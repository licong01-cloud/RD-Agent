#!/usr/bin/env python3
"""分析session文件，验证文档中的问题"""

import pickle
import sys
from pathlib import Path

def analyze_session(session_path: Path):
    """分析session文件"""
    try:
        with open(session_path, 'rb') as f:
            session = pickle.load(f)

        # 检查 session 对象的结构
        print('Session type:', type(session))
        print('Has trace:', hasattr(session, 'trace'))

        if hasattr(session, 'trace'):
            trace = session.trace
            print('Trace type:', type(trace))
            print('Has hist:', hasattr(trace, 'hist'))

            if hasattr(trace, 'hist'):
                hist = trace.hist
                print('History length:', len(hist))
                print('History types:', [type(t[0]).__name__ for t in hist[:5]])

                # 查找最后一个被接受的因子实验
                for i, (exp, feedback) in enumerate(reversed(hist)):
                    if feedback and hasattr(feedback, 'decision') and feedback.decision:
                        exp_type = type(exp).__name__
                        if 'Factor' in exp_type:
                            print(f'\nLast SOTA factor: {exp_type} at index {len(hist) - 1 - i}')
                            print('Has experiment_workspace:', hasattr(exp, 'experiment_workspace'))

                            if hasattr(exp, 'experiment_workspace'):
                                ws = exp.experiment_workspace
                                print('Workspace type:', type(ws))
                                print('Has workspace_path:', hasattr(ws, 'workspace_path'))

                                if hasattr(ws, 'workspace_path'):
                                    print('Workspace path:', ws.workspace_path)
                            break
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    session_path = Path('log/2025-12-18_16-24-29-487030/__session__/0/1_coding')
    analyze_session(session_path)
