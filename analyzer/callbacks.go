// Copyright 2023 Google LLC
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

package analyzer

import (
	"fmt"
	"go/token"
	"go/types"
	"strconv"

	"golang.org/x/tools/go/callgraph"
	"golang.org/x/tools/go/ssa"
)

// callbackPass adds call graph edges for function values which are invoked
// indirectly through reflection or through an interface method assertion.
//
// The VTA call graph is deliberately conservative for reflect: it does not
// know that a reflect.Value contains a particular function, so calls such as
// reflect.Value.Call are treated as opaque.  Similarly, when a type assertion
// to a method-set interface has no corresponding concrete type at its call
// site, VTA has no callee for the subsequent interface method call.
//
// This pass recovers the function values which are still statically visible in
// those situations.  It only adds edges; it never removes edges that VTA
// found.  SAFE functions therefore remain terminal because the capability
// search ignores them, and calls whose callbacks cannot be traced continue to
// be classified as UNANALYZED or REFLECT exactly as before.
type callbackPass struct {
	graph *callgraph.Graph

	// incoming is a snapshot of the call graph before this pass adds edges.
	// Parameter resolution uses it so synthetic edges cannot feed back into
	// the analysis.
	incoming map[*ssa.Function][]*callgraph.Edge

	// storeValues maps a memory location to values stored in it.  The values
	// are found by walking all instructions in the program.
	storeValues map[string][]ssa.Value

	// freeVarBindings maps a free variable to the values captured for it by
	// MakeClosure instructions.
	freeVarBindings map[*ssa.FreeVar][]freeVarBinding

	// added prevents duplicate synthetic edges.
	added map[functionEdge]struct{}

	// paramCache and paramActive implement memoized, cycle-safe resolution of
	// a function parameter to the function values that can reach it.
	paramCache  map[parameterRef]functionSet
	paramActive map[parameterRef]bool

	// assertionCandidates caches method implementations for an asserted
	// interface and method name.
	assertionCandidates map[string][]*ssa.Function
}

type functionEdge struct {
	caller *ssa.Function
	callee *ssa.Function
}

type parameterRef struct {
	fn    *ssa.Function
	index int
}

type freeVarBinding struct {
	value ssa.Value
	outer *ssa.Function
}

type functionSet map[*ssa.Function]struct{}
type parameterSet map[parameterRef]struct{}
type valueSubst map[ssa.Value]ssa.Value

func (s functionSet) add(fn *ssa.Function) {
	if fn != nil {
		s[fn] = struct{}{}
	}
}

func (s functionSet) merge(other functionSet) {
	for fn := range other {
		s[fn] = struct{}{}
	}
}

func (s parameterSet) add(fn *ssa.Function, index int) {
	if fn != nil {
		s[parameterRef{fn, index}] = struct{}{}
	}
}

// addCallbackEdges augments the call graph with edges for traceable function
// values passed through reflect and for unresolved interface method-set type
// assertions.
func addCallbackEdges(graph *callgraph.Graph, allFunctions map[*ssa.Function]bool) {
	p := &callbackPass{
		graph:               graph,
		incoming:            make(map[*ssa.Function][]*callgraph.Edge),
		storeValues:         make(map[string][]ssa.Value),
		freeVarBindings:     make(map[*ssa.FreeVar][]freeVarBinding),
		added:               make(map[functionEdge]struct{}),
		paramCache:          make(map[parameterRef]functionSet),
		paramActive:         make(map[parameterRef]bool),
		assertionCandidates: make(map[string][]*ssa.Function),
	}
	for fn, node := range graph.Nodes {
		p.incoming[fn] = append([]*callgraph.Edge(nil), node.In...)
	}
	for fn := range allFunctions {
		if fn == nil || fn.Blocks == nil {
			continue
		}
		p.collectMemory(fn)
	}
	p.addReflectCallbackEdges(allFunctions)
	p.addInterfaceAssertionEdges(allFunctions)
}

func (p *callbackPass) collectMemory(fn *ssa.Function) {
	for _, b := range fn.Blocks {
		for _, instr := range b.Instrs {
			switch instr := instr.(type) {
			case *ssa.Store:
				if key, ok := memoryKey(instr.Addr); ok {
					p.storeValues[key] = append(p.storeValues[key], instr.Val)
				}
			case *ssa.MakeClosure:
				closure, ok := instr.Fn.(*ssa.Function)
				if !ok {
					continue
				}
				for i, fv := range closure.FreeVars {
					if i >= len(instr.Bindings) {
						break
					}
					p.freeVarBindings[fv] = append(p.freeVarBindings[fv], freeVarBinding{
						value: instr.Bindings[i],
						outer: fn,
					})
				}
			}
		}
	}
}

// addReflectCallbackEdges finds calls to reflect.Value.Call and similar
// operations, traces the function value held by the receiver, and adds edges
// from the calling function to the traced function.
func (p *callbackPass) addReflectCallbackEdges(allFunctions map[*ssa.Function]bool) {
	for fn := range allFunctions {
		if fn == nil || fn.Blocks == nil {
			continue
		}
		if !isAnalyzedCallbackFunction(fn) {
			// Reflection in the standard library is already covered by its
			// explicit capability classifications.  Following its internal
			// callback plumbing would add noise without improving dependency
			// attribution.
			continue
		}
		for _, b := range fn.Blocks {
			for _, instr := range b.Instrs {
				call, ok := instr.(*ssa.Call)
				if !ok {
					continue
				}
				source := reflectCallbackSource(call)
				if source == nil {
					continue
				}
				funcs, params := p.traceValue(source, make(map[ssa.Value]bool))
				p.applyTrace(fn, call, funcs, params)
			}
		}
	}
}

// reflectCallbackSource returns the SSA value whose contents may be invoked by
// the reflect operation represented by call.  It returns nil for calls that
// do not invoke a function value.
func reflectCallbackSource(call *ssa.Call) ssa.Value {
	callee := call.Common().StaticCallee()
	if callee == nil || packagePath(callee) != "reflect" {
		return nil
	}
	args := call.Common().Args
	if callee.Signature == nil {
		return nil
	}
	if recv := callee.Signature.Recv(); recv != nil {
		if !isReflectValueType(recv.Type()) || len(args) == 0 {
			return nil
		}
		switch callee.Name() {
		case "Call", "CallSlice":
			// The receiver is the reflect.Value containing the function to call.
			return args[0]
		case "Set":
			// The assigned value may be a function value.
			if len(args) >= 2 {
				return args[1]
			}
		case "SetMapIndex":
			if len(args) >= 3 {
				return args[2]
			}
		}
		return nil
	}
	switch callee.Name() {
	case "MakeFunc":
		// The second argument is the function which implements the created
		// function value.
		if len(args) >= 2 {
			return args[1]
		}
	}
	return nil
}

// applyTrace adds edges from caller to the functions in funcs.  Parameters in
// params are resolved through their function's call sites; this also covers
// parameters of an outer function when the callback arrives through a closure.
func (p *callbackPass) applyTrace(caller *ssa.Function, site ssa.CallInstruction, funcs functionSet, params parameterSet) {
	for fn := range funcs {
		p.addEdge(caller, fn, site)
	}
	for param := range params {
		p.resolveParameterFuncs(param).forEach(func(fn *ssa.Function) {
			p.addEdge(caller, fn, site)
		})
	}
}

func (s functionSet) forEach(fn func(*ssa.Function)) {
	for f := range s {
		fn(f)
	}
}

func (p *callbackPass) addEdge(caller, callee *ssa.Function, site ssa.CallInstruction) {
	if caller == nil || callee == nil {
		return
	}
	key := functionEdge{caller, callee}
	if _, ok := p.added[key]; ok {
		return
	}
	callerNode := p.graph.Nodes[caller]
	calleeNode := p.graph.Nodes[callee]
	if callerNode == nil {
		callerNode = p.graph.CreateNode(caller)
	}
	if calleeNode == nil {
		calleeNode = p.graph.CreateNode(callee)
	}
	callgraph.AddEdge(callerNode, site, calleeNode)
	p.added[key] = struct{}{}
}

// resolveParameterFuncs returns the function values which can be passed for
// parameter index of fn at any of fn's call sites.
func (p *callbackPass) resolveParameterFuncs(ref parameterRef) functionSet {
	if cached, ok := p.paramCache[ref]; ok {
		return cached
	}
	if p.paramActive[ref] {
		// Break cycles in recursive wrappers.
		return make(functionSet)
	}
	p.paramActive[ref] = true
	defer delete(p.paramActive, ref)

	result := make(functionSet)
	for _, edge := range p.incoming[ref.fn] {
		site := edge.Site
		if site == nil {
			continue
		}
		arg := callArgumentForParameter(site, ref.fn, ref.index)
		if arg == nil {
			continue
		}
		funcs, params := p.traceValue(arg, make(map[ssa.Value]bool))
		result.merge(funcs)
		for param := range params {
			result.merge(p.resolveParameterFuncs(param))
		}
	}
	p.paramCache[ref] = result
	return result
}

// callArgumentForParameter returns the argument value corresponding to
// parameter index of callee at site.
func callArgumentForParameter(site ssa.CallInstruction, callee *ssa.Function, index int) ssa.Value {
	cc := site.Common()
	if cc.IsInvoke() {
		// In invoke mode the receiver is Value, and Args contains only the
		// explicit arguments.
		if index == 0 {
			return cc.Value
		}
		if i := index - 1; i >= 0 && i < len(cc.Args) {
			return cc.Args[i]
		}
		return nil
	}
	// A static method call's Args slice includes the receiver.  A method
	// value's bound receiver is stored in the closure, so it is not in Args.
	if callee.Signature != nil && callee.Signature.Recv() != nil && cc.StaticCallee() != callee {
		if index == 0 {
			return nil
		}
		if i := index - 1; i >= 0 && i < len(cc.Args) {
			return cc.Args[i]
		}
		return nil
	}
	if index >= 0 && index < len(cc.Args) {
		return cc.Args[index]
	}
	return nil
}

// traceValue finds function values and function-typed parameters which can
// flow to v.  It follows the SSA value, interface conversions, local memory,
// and closure bindings.  It intentionally does not attempt a full points-to
// analysis; untraceable values simply produce no result.
func (p *callbackPass) traceValue(v ssa.Value, stack map[ssa.Value]bool) (functionSet, parameterSet) {
	return p.traceValueWithSubst(v, stack, nil)
}

func (p *callbackPass) traceValueWithSubst(v ssa.Value, stack map[ssa.Value]bool, subst valueSubst) (functionSet, parameterSet) {
	funcs := make(functionSet)
	params := make(parameterSet)
	if v == nil {
		return funcs, params
	}
	if replacement, ok := subst[v]; ok {
		return p.traceValueWithSubst(replacement, stack, nil)
	}
	if stack[v] {
		return funcs, params
	}
	stack[v] = true
	defer delete(stack, v)

	switch v := v.(type) {
	case *ssa.Function:
		funcs.add(v)
	case *ssa.MakeClosure:
		if fn, ok := v.Fn.(*ssa.Function); ok {
			funcs.add(fn)
		}
	case *ssa.Parameter:
		if index, ok := parameterIndex(v); ok {
			params.add(v.Parent(), index)
		}
	case *ssa.FreeVar:
		for _, binding := range p.freeVarBindings[v] {
			boundFuncs, boundParams := p.traceValueWithSubst(binding.value, stack, subst)
			funcs.merge(boundFuncs)
			for param := range boundParams {
				if param.fn == nil {
					param.fn = binding.outer
				}
				params.add(param.fn, param.index)
			}
		}
	case *ssa.MakeInterface:
		return p.traceValueWithSubst(v.X, stack, subst)
	case *ssa.ChangeInterface:
		return p.traceValueWithSubst(v.X, stack, subst)
	case *ssa.ChangeType:
		return p.traceValueWithSubst(v.X, stack, subst)
	case *ssa.Convert:
		return p.traceValueWithSubst(v.X, stack, subst)
	case *ssa.TypeAssert:
		return p.traceValueWithSubst(v.X, stack, subst)
	case *ssa.Phi:
		for _, edge := range v.Edges {
			edgeFuncs, edgeParams := p.traceValueWithSubst(edge, stack, subst)
			funcs.merge(edgeFuncs)
			for param := range edgeParams {
				params.add(param.fn, param.index)
			}
		}
	case *ssa.UnOp:
		if v.Op == token.MUL {
			if key, ok := memoryKey(v.X); ok {
				for _, stored := range p.storeValues[key] {
					storedFuncs, storedParams := p.traceValueWithSubst(stored, stack, subst)
					funcs.merge(storedFuncs)
					for param := range storedParams {
						params.add(param.fn, param.index)
					}
				}
			}
		}
	case *ssa.Field:
		return p.traceValueWithSubst(v.X, stack, subst)
	case *ssa.FieldAddr:
		return p.traceValueWithSubst(v.X, stack, subst)
	case *ssa.Index:
		return p.traceValueWithSubst(v.X, stack, subst)
	case *ssa.IndexAddr:
		return p.traceValueWithSubst(v.X, stack, subst)
	case *ssa.Lookup:
		return p.traceValueWithSubst(v.X, stack, subst)
	case *ssa.Slice:
		return p.traceValueWithSubst(v.X, stack, subst)
	case *ssa.Range:
		return p.traceValueWithSubst(v.X, stack, subst)
	case *ssa.Next:
		if it, ok := v.Iter.(*ssa.Range); ok {
			return p.traceValueWithSubst(it.X, stack, subst)
		}
	case *ssa.Extract:
		if call, ok := v.Tuple.(*ssa.Call); ok {
			return p.traceCallResult(call, v.Index, stack, subst)
		}
		return p.traceValueWithSubst(v.Tuple, stack, subst)
	case *ssa.Call:
		if funcs, params := p.traceReflectValueSource(v, stack, subst); len(funcs) != 0 || len(params) != 0 {
			return funcs, params
		}
		return p.traceCallResult(v, -1, stack, subst)
	}
	return funcs, params
}

// traceReflectValueSource handles calls whose result is a reflect.Value that
// contains a function value, or whose result is a function value.
func (p *callbackPass) traceReflectValueSource(call *ssa.Call, stack map[ssa.Value]bool, subst valueSubst) (functionSet, parameterSet) {
	emptyFuncs := make(functionSet)
	emptyParams := make(parameterSet)
	callee := call.Common().StaticCallee()
	if callee == nil || callee.Signature == nil {
		return emptyFuncs, emptyParams
	}
	args := call.Common().Args
	if packagePath(callee) == "reflect" {
		if callee.Signature.Recv() == nil {
			switch callee.Name() {
			case "ValueOf":
				if len(args) >= 1 {
					return p.traceValueWithSubst(args[0], stack, subst)
				}
			case "MakeFunc":
				if len(args) >= 2 {
					return p.traceValueWithSubst(args[1], stack, subst)
				}
			}
			return emptyFuncs, emptyParams
		}
		if isReflectValueType(callee.Signature.Recv().Type()) && len(args) >= 1 {
			switch callee.Name() {
			case "Elem", "Field", "FieldByIndex", "FieldByName", "Index",
				"Interface", "Convert", "Method", "MethodByName":
				return p.traceValueWithSubst(args[0], stack, subst)
			}
		}
	}
	return emptyFuncs, emptyParams
}

// traceCallResult follows a function's return values back to the arguments of
// call.  This recovers simple generic and interface wrappers which return a
// function value, for example an identity function used before reflect.ValueOf.
func (p *callbackPass) traceCallResult(call *ssa.Call, index int, stack map[ssa.Value]bool, subst valueSubst) (functionSet, parameterSet) {
	funcs := make(functionSet)
	params := make(parameterSet)
	callee := call.Common().StaticCallee()
	if callee == nil || callee.Blocks == nil {
		return funcs, params
	}
	callSubst := callSubstitution(call, callee)
	for _, b := range callee.Blocks {
		if len(b.Instrs) == 0 {
			continue
		}
		ret, ok := b.Instrs[len(b.Instrs)-1].(*ssa.Return)
		if !ok {
			continue
		}
		if index >= 0 {
			if index >= len(ret.Results) {
				continue
			}
			retFuncs, retParams := p.traceValueWithSubst(ret.Results[index], stack, callSubst)
			funcs.merge(retFuncs)
			for param := range retParams {
				params.add(param.fn, param.index)
			}
			continue
		}
		if len(ret.Results) != 1 {
			continue
		}
		retFuncs, retParams := p.traceValueWithSubst(ret.Results[0], stack, callSubst)
		funcs.merge(retFuncs)
		for param := range retParams {
			params.add(param.fn, param.index)
		}
	}
	return funcs, params
}

// callSubstitution maps the parameters and free variables of callee to the
// values supplied at call.
func callSubstitution(call *ssa.Call, callee *ssa.Function) valueSubst {
	subst := make(valueSubst)
	for i, param := range callee.Params {
		if arg := callArgumentForParameter(call, callee, i); arg != nil {
			subst[param] = arg
		}
	}
	if mc, ok := call.Common().Value.(*ssa.MakeClosure); ok {
		if fn, ok := mc.Fn.(*ssa.Function); ok && fn == callee {
			for i, freeVar := range callee.FreeVars {
				if i < len(mc.Bindings) {
					subst[freeVar] = mc.Bindings[i]
				}
			}
		}
	}
	return subst
}

func parameterIndex(p *ssa.Parameter) (int, bool) {
	parent := p.Parent()
	if parent == nil {
		return 0, false
	}
	for i, param := range parent.Params {
		if param == p {
			return i, true
		}
	}
	return 0, false
}

// memoryKey returns a stable key for a memory location.  Locals are keyed by
// object identity; globals and fields of globals use their object identity so
// stores in init functions and stores in other functions agree.
func memoryKey(v ssa.Value) (string, bool) {
	switch v := v.(type) {
	case *ssa.Global:
		return "global:" + v.String(), true
	case *ssa.Alloc:
		return fmt.Sprintf("alloc:%p", v), true
	case *ssa.FieldAddr:
		key, ok := memoryKey(v.X)
		if !ok {
			return "", false
		}
		return key + ".field" + strconv.Itoa(v.Field), true
	case *ssa.IndexAddr:
		key, ok := memoryKey(v.X)
		if !ok {
			return "", false
		}
		// We do not distinguish indexes.  This is conservative and avoids
		// missing a callback stored in an array or slice.
		return key + ".index", true
	case *ssa.Slice:
		return memoryKey(v.X)
	case *ssa.UnOp:
		if v.Op == token.MUL {
			return memoryKey(v.X)
		}
	case *ssa.Parameter:
		if index, ok := parameterIndex(v); ok {
			return fmt.Sprintf("param:%p:%d", v.Parent(), index), true
		}
	}
	return "", false
}

func isReflectValueType(t types.Type) bool {
	if t == nil {
		return false
	}
	if ptr, ok := types.Unalias(t).(*types.Pointer); ok {
		t = ptr.Elem()
	}
	named, ok := types.Unalias(t).(*types.Named)
	if !ok || named.Obj() == nil || named.Obj().Pkg() == nil {
		return false
	}
	return named.Obj().Pkg().Path() == "reflect" && named.Obj().Name() == "Value"
}

// addInterfaceAssertionEdges handles interface method calls whose receiver is
// a type assertion to a method-set interface.  VTA resolves these calls when
// it can see a concrete dynamic type in the program.  When it cannot, we add
// edges to every method implementation whose receiver implements the asserted
// interface.  This makes an exported helper such as MaybeChmod show the
// capability it can invoke instead of appearing indistinguishable from code
// which cannot call Chmod at all.
func (p *callbackPass) addInterfaceAssertionEdges(allFunctions map[*ssa.Function]bool) {
	for fn, node := range p.graph.Nodes {
		if fn == nil || fn.Blocks == nil {
			continue
		}
		if !isAnalyzedCallbackFunction(fn) {
			continue
		}
		if fn.TypeParams() != nil && fn.TypeParams().Len() > 0 && len(fn.TypeArgs()) == 0 {
			// The generic origin is not the function which runs at a call site.
			// Its instantiations are analyzed separately; adding edges here
			// would create duplicate, shape-dependent call paths.
			continue
		}
		for _, b := range fn.Blocks {
			for _, instr := range b.Instrs {
				call, ok := instr.(*ssa.Call)
				if !ok {
					continue
				}
				cc := call.Common()
				if !cc.IsInvoke() || cc.Method == nil {
					continue
				}
				iface := p.traceAssertedInterface(cc.Value, make(map[ssa.Value]bool), nil)
				if iface == nil {
					continue
				}
				if hasEdgeForSite(node, call) {
					// VTA already found at least one implementation.
					continue
				}
				methodName := cc.Method.Name()
				cacheKey := methodName + "|" + types.TypeString(iface, func(pkg *types.Package) string {
					return pkg.Path()
				})
				candidates, ok := p.assertionCandidates[cacheKey]
				if !ok {
					for candidate := range allFunctions {
						if candidate == nil || candidate.Signature == nil {
							continue
						}
						if candidate.Synthetic != "" {
							// Promoted-method wrappers and other synthetic
							// functions do not represent distinct dynamic types.
							// The method they forward to is considered
							// separately.
							continue
						}
						if candidate.TypeParams() != nil && candidate.TypeParams().Len() > 0 &&
							len(candidate.TypeArgs()) == 0 {
							// The uninstantiated generic method is not a
							// runtime implementation; its instantiations are
							// considered separately.
							continue
						}
						recv := candidate.Signature.Recv()
						if recv == nil || candidate.Name() != methodName {
							continue
						}
						recvType := recv.Type()
						if !types.Implements(recvType, iface) &&
							!types.Implements(types.NewPointer(recvType), iface) {
							continue
						}
						candidates = append(candidates, candidate)
					}
					p.assertionCandidates[cacheKey] = candidates
				}
				for _, candidate := range candidates {
					p.addEdge(fn, candidate, call)
				}
			}
		}
	}
}

// isAnalyzedCallbackFunction reports whether callbacks in fn should be traced.
// Functions in the standard library keep their existing classifications;
// tracing them would duplicate paths which are intentionally summarized by
// the capability map.
func isAnalyzedCallbackFunction(fn *ssa.Function) bool {
	pkg := functionPackagePath(fn)
	return pkg == "" || !isStdLib(pkg)
}

func functionPackagePath(fn *ssa.Function) string {
	for f := fn; f != nil; f = f.Parent() {
		if pkg := packagePath(f); pkg != "" {
			return pkg
		}
	}
	return ""
}

func hasEdgeForSite(node *callgraph.Node, site ssa.CallInstruction) bool {
	for _, edge := range node.Out {
		if edge.Site == site {
			return true
		}
	}
	return false
}

// traceAssertedInterface returns the interface type of a type assertion which
// produces v, if v is the value produced by such an assertion.
func (p *callbackPass) traceAssertedInterface(v ssa.Value, stack map[ssa.Value]bool, subst valueSubst) *types.Interface {
	if v == nil {
		return nil
	}
	if replacement, ok := subst[v]; ok {
		return p.traceAssertedInterface(replacement, stack, nil)
	}
	if stack[v] {
		return nil
	}
	stack[v] = true
	defer delete(stack, v)
	switch v := v.(type) {
	case *ssa.TypeAssert:
		iface, ok := types.Unalias(v.AssertedType).Underlying().(*types.Interface)
		if ok {
			return iface
		}
	case *ssa.Extract:
		if call, ok := v.Tuple.(*ssa.Call); ok {
			return p.traceAssertedCallResult(call, v.Index, stack, subst)
		}
		return p.traceAssertedInterface(v.Tuple, stack, subst)
	case *ssa.MakeInterface:
		return p.traceAssertedInterface(v.X, stack, subst)
	case *ssa.ChangeInterface:
		return p.traceAssertedInterface(v.X, stack, subst)
	case *ssa.ChangeType:
		return p.traceAssertedInterface(v.X, stack, subst)
	case *ssa.Convert:
		return p.traceAssertedInterface(v.X, stack, subst)
	case *ssa.Phi:
		for _, edge := range v.Edges {
			if iface := p.traceAssertedInterface(edge, stack, subst); iface != nil {
				return iface
			}
		}
	case *ssa.UnOp:
		if v.Op == token.MUL {
			if key, ok := memoryKey(v.X); ok {
				for _, stored := range p.storeValues[key] {
					if iface := p.traceAssertedInterface(stored, stack, subst); iface != nil {
						return iface
					}
				}
			}
		}
	case *ssa.Field:
		return p.traceAssertedInterface(v.X, stack, subst)
	case *ssa.FieldAddr:
		return p.traceAssertedInterface(v.X, stack, subst)
	case *ssa.Index:
		return p.traceAssertedInterface(v.X, stack, subst)
	case *ssa.IndexAddr:
		return p.traceAssertedInterface(v.X, stack, subst)
	case *ssa.Lookup:
		return p.traceAssertedInterface(v.X, stack, subst)
	case *ssa.Slice:
		return p.traceAssertedInterface(v.X, stack, subst)
	case *ssa.Call:
		return p.traceAssertedCallResult(v, -1, stack, subst)
	}
	return nil
}

// traceAssertedCallResult follows the return values of a call to an interface
// type assertion.  It is the interface analogue of traceCallResult and
// supports helpers which return the result of a type assertion.
func (p *callbackPass) traceAssertedCallResult(call *ssa.Call, index int, stack map[ssa.Value]bool, subst valueSubst) *types.Interface {
	callee := call.Common().StaticCallee()
	if callee == nil || callee.Blocks == nil {
		return nil
	}
	callSubst := callSubstitution(call, callee)
	for _, b := range callee.Blocks {
		if len(b.Instrs) == 0 {
			continue
		}
		ret, ok := b.Instrs[len(b.Instrs)-1].(*ssa.Return)
		if !ok {
			continue
		}
		if index >= 0 {
			if index < len(ret.Results) {
				if iface := p.traceAssertedInterface(ret.Results[index], stack, callSubst); iface != nil {
					return iface
				}
			}
			continue
		}
		if len(ret.Results) == 1 {
			if iface := p.traceAssertedInterface(ret.Results[0], stack, callSubst); iface != nil {
				return iface
			}
		}
	}
	return nil
}
