import re
from pathlib import Path
import pytest
import torch

from chebyshev.exporter import export_to_c_header
from chebyshev.models import StudentModel


def test_export_c_header_structure_and_constants(tmp_path: Path):
    num_channels = 4
    order = 6
    hidden_dim = 16
    out_features = 1
    
    model = StudentModel(
        num_channels=num_channels,
        order=order,
        hidden_dim=hidden_dim,
        out_features=out_features,
    )
    
    output_file = tmp_path / "chebyshev_engine.h"
    content = export_to_c_header(model, output_file)
    
    assert output_file.exists()
    assert len(content) > 0
    
    # Check guard macros
    assert "#ifndef CHEBYSHEV_ENGINE_H" in content
    assert "#define CHEBYSHEV_ENGINE_H" in content
    assert "#endif /* CHEBYSHEV_ENGINE_H */" in content
    
    # Check standard headers for strict-aliasing-safe IEEE-754 bit check
    assert "#include <stdint.h>" in content
    assert "#include <string.h>" in content

    # Check extern C guards
    assert '#ifdef __cplusplus\nextern "C" {' in content
    assert '#ifdef __cplusplus\n}' in content

    # Check dimension macros
    assert re.search(rf"#define CHEB_NUM_CHANNELS\s+{num_channels}\b", content) is not None
    assert re.search(rf"#define CHEB_ORDER\s+{order}\b", content) is not None
    assert re.search(rf"#define CHEB_HIDDEN_DIM\s+{hidden_dim}\b", content) is not None
    assert re.search(rf"#define CHEB_NUM_OUTPUTS\s+{out_features}\b", content) is not None
    assert re.search(rf"#define CHEB_PERSISTENT_STATE_BYTES\s+{num_channels * order * 4}\b", content) is not None
    assert re.search(rf"#define CHEB_STACK_SCRATCH_BYTES\s+{(order + hidden_dim) * 4}\b", content) is not None

    # Check Checksum macro
    assert re.search(r"#define CHEB_MODEL_CHECKSUM\s+0x[0-9A-Fa-f]{8}U", content) is not None

    # Check static arrays
    assert "static const float CHEB_A[CHEB_ORDER][CHEB_ORDER]" in content
    assert "static const float CHEB_B[CHEB_ORDER]" in content
    assert "static const float CHEB_MEAN[CHEB_NUM_CHANNELS]" in content
    assert "static const float CHEB_INV_STD[CHEB_NUM_CHANNELS]" in content
    assert "static const float CHEB_W1[CHEB_HIDDEN_DIM][CHEB_NUM_CHANNELS * CHEB_ORDER]" in content
    assert "static const float CHEB_B1[CHEB_HIDDEN_DIM]" in content
    assert "static const float CHEB_W2[CHEB_NUM_OUTPUTS][CHEB_HIDDEN_DIM]" in content
    assert "static const float CHEB_B2[CHEB_NUM_OUTPUTS]" in content

    # Check static inline functions (ADR-0006) and -ffast-math-safe IEEE-754 bit quarantine
    assert "static inline void chebyshev_init" in content
    assert "static inline void chebyshev_reset" in content
    assert "static inline void chebyshev_step" in content
    assert "static inline float chebyshev_sanitize" in content
    assert "memcpy(&bits, &x, sizeof(bits));" in content
    assert "0x7F800000U" in content

    # Check anti in-place overwrite buffer
    assert "float next_s[CHEB_ORDER];" in content

    # Check accurate memory footprint breakdown and absence of uncertified MISRA-C claims
    assert "Persistent Recurrent State RAM" in content
    assert "Active Call-Stack Scratch" in content
    assert "Total Estimated Runtime SRAM" in content
    assert "Estimated Flash/ROM footprint" in content
    assert "MISRA-C" not in content


def test_float_literal_syntax(tmp_path: Path):
    model = StudentModel(num_channels=2, order=2, hidden_dim=4, out_features=1)
    output_file = tmp_path / "chebyshev_engine.h"
    content = export_to_c_header(model, output_file)
    
    # Find all float literals inside array initializers (tokens ending in 'f')
    float_matches = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?f", content)
    assert len(float_matches) > 10
    
    # Validate each float literal: must have either a decimal point or exponent before 'f'
    for literal in float_matches:
        base = literal[:-1]  # strip trailing 'f'
        assert "." in base or "e" in base or "E" in base, f"Invalid C99 float literal: {literal}"
