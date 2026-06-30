import numpy as np

from PINN_funcs.data_handling import load_npz_bundle


def _write_valid_bundle(path):
    X_data = np.array([[0.0, 0.1], [0.5, 0.2], [1.0, 0.3]], dtype=np.float32)
    U_data = np.array([[1.0, 2.0], [1.2, 2.2], [1.4, 2.4]], dtype=np.float32)
    X_val = np.array([[0.2, 0.2], [0.8, 0.25]], dtype=np.float32)
    X_BC_inlet = np.array([[0.0, 0.1], [0.0, 0.2]], dtype=np.float32)
    U_BC_inlet = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    U_val = {
        "u": np.array([1.0, 1.1], dtype=np.float32),
        "v": np.array([0.0, 0.1], dtype=np.float32),
    }   

    np.savez(
        path,
        X_data=X_data,
        U_data=U_data,
        X_val=X_val,
        U_val=U_val,
        X_BC={"inlet": X_BC_inlet},
        U_BC={"inlet": U_BC_inlet}
    )


def test_load_case_bundle_success(tmp_path):
    bundle_path = tmp_path / "case_bundle.npz"
    _write_valid_bundle(bundle_path)

    out = load_npz_bundle(
        str(bundle_path),
        dtype="float32",
        BC_names=("inlet",)
    )

    assert out["X_data"].shape == (3, 2)
    assert out["U_data"].shape == (3, 2)
    assert out["X_val"].shape == (2, 2)
    assert set(out["X_BC"].keys()) == {"inlet"}
    assert set(out["U_BC"].keys()) == {"inlet"}
    assert out["lb"].shape == (2,)
    assert out["ub"].shape == (2,)
    np.testing.assert_allclose(out["lb"], np.array([0.0, 0.1], dtype=np.float32))
    np.testing.assert_allclose(out["ub"], np.array([1.0, 0.3], dtype=np.float32))

