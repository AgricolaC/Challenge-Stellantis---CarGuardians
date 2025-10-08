from challenge.utils.paths import project_root, data_root

def test_paths_exist():
    assert project_root().exists()
    assert data_root().exists()
