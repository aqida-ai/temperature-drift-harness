"""
AQIDA_CONSTITUTIONAL_SCOPE: diagnostic
THEORY_BINDING_MAP:
- CORE_LOGIC: storage width preserves token values, not a scientific outcome.
- elm_eulerian: inspect the actual exported analyzer's input predicate.
- n_s_u: artificial falsification cases distinguish compatible and invalid input.
CLAIM_BOUNDARY: CPU software fixture only; no model, corpus or GPU measurement.
FALSIFICATION_TESTS: int32/int64 acceptance; mixed, float, bool, unsigned, short
integer and malformed prompt-grid rejection; no CUDA initialization.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
from . import analyze as A


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve prior fixture receipts.'
    tree = ast.parse(inspect.getsource(A.audit_run))
    nodes = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and len(n.args)==2 and
             isinstance(n.func,ast.Attribute) and n.func.attr=='check' and
             isinstance(n.args[1],ast.Constant) and n.args[1].value=='Prepared context grid changed.']
    assert len(nodes)==1
    predicate = compile(ast.Expression(nodes[0].args[0]),'<exported audit input predicate>','eval')
    a = np.arange(128*64,dtype=np.int32).reshape(128,64)
    b = np.arange(128*192,dtype=np.int32).reshape(128,192)
    def accepts(x,y):
        return bool(eval(predicate,{'np':np,'contexts':x,'quality_examples':y,
                                   'config':dict(contexts=128,prompt_tokens=64)}))
    cases = [('signed int32 accepted',accepts(a,b)),
             ('signed int64 accepted',accepts(a.astype(np.int64),b.astype(np.int64))),
             ('mixed width rejected',not accepts(a,b.astype(np.int64))),
             ('short prompt grid rejected',not accepts(a[:127],b))]
    for dtype in [np.float64,np.float32,np.bool_,np.uint32,np.int16]:
        cases.append((np.dtype(dtype).name+' rejected',not accepts(a.astype(dtype),b.astype(dtype))))
    checks = [dict(name=name,passed=bool(ok)) for name,ok in cases]
    import torch
    checks.append(dict(name='CUDA remains uninitialized',passed=not torch.cuda.is_initialized()))
    result = dict(status='PASS_PORTABLE_STORAGE_DTYPE_SELF_TEST' if all(r['passed'] for r in checks) else 'FAIL',
        passed=sum(r['passed'] for r in checks),total=len(checks),checks=checks,
        source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ['analyze.py','storage_dtype_self_test.py']},
        fixture_only=True,no_external_data_or_model=True,CUDA_initialized=torch.cuda.is_initialized())
    with args.output.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'checks','source_sha256'}},indent=2))
    assert result['status']=='PASS_PORTABLE_STORAGE_DTYPE_SELF_TEST'


if __name__=='__main__':
    main()
