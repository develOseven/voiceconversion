import json
import os

import pytest

from voiceconversion.data.imported_model_info import (
    RVCImportedModelInfo,
    load_all_imported_model_infos,
    load_imported_model_info,
    save_imported_model_info,
)


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary directory for model slots."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


# Tests for load_imported_model_info


def test_loadimportedmodelinfo_no_file(temp_model_dir):
    assert load_imported_model_info(0, temp_model_dir / "0") is None


def test_loadimportedmodelinfo_non_rvc_json(temp_model_dir):
    storage_dir = temp_model_dir / "0"
    storage_dir.mkdir()
    json_data = {"voiceChangerType": "None", "name": "TestSlot"}
    with open(storage_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f)

    assert load_imported_model_info(0, temp_model_dir / "0") is None


def test_loadimportedmodelinfo_loads_rvc_json(temp_model_dir):
    storage_dir = temp_model_dir / "1"
    storage_dir.mkdir()
    json_data = {
        "voiceChangerType": "RVC",
        "name": "RVCSlot",
        "modelFile": "model.pth",
        "modelFileOnnx": "model.onnx",
        "samplingRate": 44100,
    }
    with open(storage_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f)

    info = load_imported_model_info(1, temp_model_dir / "1")
    assert info.name == "RVCSlot"
    assert info.modelFile == "model.pth"
    assert info.samplingRate == 44100


# Tests for load_all_imported_model_infos


def test_loadallimportedmodelinfos_empty(temp_model_dir):
    infos = load_all_imported_model_infos(str(temp_model_dir))
    assert not infos


def test_loadallimportedmodelinfos_loads(temp_model_dir):
    for i in range(3):
        storage_dir = temp_model_dir / str(i)
        storage_dir.mkdir()
        with open(storage_dir / "params.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "voiceChangerType": "RVC",
                    "name": f"Model {i}",
                    "modelFile": f"{i}.pth",
                },
                f,
            )

    infos = load_all_imported_model_infos(str(temp_model_dir))

    assert len(infos) == 3
    info0 = infos[0]
    assert type(info0) is RVCImportedModelInfo
    assert info0.name == "Model 0"
    assert info0.storageDir == str(temp_model_dir / "0")
    assert info0.modelFile == "0.pth"
    info2 = infos[2]
    assert type(info0) is RVCImportedModelInfo
    assert info2.name == "Model 2"
    assert info2.storageDir == str(temp_model_dir / "2")
    assert info2.modelFile == "2.pth"


def test_loadallimportedmodelinfos_gaps(temp_model_dir):
    storage_dir = temp_model_dir / "3"
    storage_dir.mkdir()
    with open(storage_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "voiceChangerType": "RVC",
                "name": "Model 3",
                "modelFile": "3.pth",
            },
            f,
        )

    infos = load_all_imported_model_infos(str(temp_model_dir))

    assert len(infos) == 1
    assert infos[3].name == "Model 3"


# Tests for save_imported_model_info


def test_saveimportedmodelinfo_creates_json_file(temp_model_dir):
    info = RVCImportedModelInfo(
        storageDir=temp_model_dir / "2", name="SaveTest", modelFile="abc.pth"
    )

    save_imported_model_info(info)

    json_path = temp_model_dir / "2" / "params.json"
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "storageDir" not in data
    assert data["name"] == "SaveTest"
    assert data["modelFile"] == "abc.pth"
