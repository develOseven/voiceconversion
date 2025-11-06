import json
import os
from pathlib import Path

import pytest

from voiceconversion.data.imported_model_info import (
    ImportedModelInfo,
    RVCImportedModelInfo,
    load_all_imported_model_infos,
    save_imported_model_info,
)
from voiceconversion.imported_model_info_manager import ImportedModelInfoManager


@pytest.fixture
def model_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    return str(d)


def make_info(model_dir: str, id: int, name="model"):
    """Helper to create a fake ImportedModelInfo and persist it."""
    storage_dir = os.path.join(model_dir, str(id))
    os.makedirs(storage_dir, exist_ok=True)
    info = RVCImportedModelInfo(
        id=id,
        storageDir=storage_dir,
        name=f"{name}-{id}",
        voiceChangerType="RVC",
    )
    save_imported_model_info(info)
    return info


def test_init_loads_existing_models(model_dir):
    # Prepare two stored models
    info0 = make_info(model_dir, 0)
    info1 = make_info(model_dir, 1)

    # Manager should load both
    mgr = ImportedModelInfoManager(model_dir)
    assert set(mgr.infos.keys()) == {0, 1}
    assert mgr.get(0).name == info0.name
    assert mgr.get(1).name == info1.name


def test_save_updates_infos(model_dir):
    mgr = ImportedModelInfoManager(model_dir)
    id, storage_dir = mgr.new_id()
    info = RVCImportedModelInfo(
        id=id,
        storageDir=storage_dir,
        name="SavedModel",
        voiceChangerType="RVC",
    )

    mgr.save(info)

    # After save, params.json should exist and infos should reload
    json_file = os.path.join(storage_dir, "params.json")
    assert os.path.exists(json_file)

    with open(json_file) as f:
        data = json.load(f)
    assert data["name"] == "SavedModel"

    # Manager should have the new id in its infos dict
    assert id in mgr.infos
    assert mgr.get(id).name == "SavedModel"


def test_get_returns_none_for_missing_id(model_dir):
    mgr = ImportedModelInfoManager(model_dir)
    assert mgr.get(12345) is None


def test_new_id_increments_based_on_existing(model_dir):
    for i in range(3):
        make_info(model_dir, i)
    mgr = ImportedModelInfoManager(model_dir)

    new_id, new_dir = mgr.new_id()
    assert new_id == 3
    assert os.path.basename(new_dir) == "3"
    assert new_dir.startswith(model_dir)


def test_new_id_returns_zero_when_empty(model_dir):
    mgr = ImportedModelInfoManager(model_dir)
    id, storage_dir = mgr.new_id()
    assert id == 0
    assert storage_dir.endswith("0")


def test_remove_deletes_storage_dir_and_removes_from_infos(model_dir):
    info = make_info(model_dir, 0)
    mgr = ImportedModelInfoManager(model_dir)
    assert mgr.get(0) is not None
    assert os.path.exists(info.storageDir)

    mgr.remove(0)
    assert mgr.get(0) is None
    assert not os.path.exists(info.storageDir)


def test_remove_ignores_missing_id(model_dir):
    mgr = ImportedModelInfoManager(model_dir)
    # Should not raise
    mgr.remove(999)


def test_save_and_reload_rvc_info_integrity(model_dir):
    info = RVCImportedModelInfo(
        id=5,
        storageDir=os.path.join(model_dir, "5"),
        name="Fancy",
        description="Test model",
        voiceChangerType="RVC",
        credit="Tester",
        termsOfUseUrl="https://example.com/terms",
    )
    save_imported_model_info(info)

    mgr = ImportedModelInfoManager(model_dir)
    loaded = mgr.get(5)
    assert loaded is not None
    assert loaded.name == "Fancy"
    assert loaded.voiceChangerType == "RVC"
    assert loaded.description == "Test model"
    assert loaded.credit == "Tester"
