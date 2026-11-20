import subprocess
from chebyshev.exporter import export_to_c_header
from chebyshev.models import StudentModel
import tempfile
from pathlib import Path

model = StudentModel(num_channels=14, order=8, hidden_dim=32, out_features=1)
with tempfile.TemporaryDirectory() as tmpdir:
    header_path = Path(tmpdir) / 'chebyshev_engine.h'
    export_to_c_header(model, header_path)
    
    c_file = Path(tmpdir) / 'test_stack.c'
    src = """#include "chebyshev_engine.h"
void run_step(chebyshev_state_t *st, const float *raw, float *out) {
    chebyshev_step(st, raw, out);
}
"""
    c_file.write_text(src, encoding='utf-8')
    
    cmd = ['gcc', '-std=c99', '-O3', '-fstack-usage', f'-I{tmpdir}', '-c', str(c_file), '-o', str(Path(tmpdir) / 'test_stack.o')]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print('Error:', res.stderr)
    else:
        su_file = Path(tmpdir) / 'test_stack.su'
        if su_file.exists():
            print('Stack usage file contents:\n' + su_file.read_text())
