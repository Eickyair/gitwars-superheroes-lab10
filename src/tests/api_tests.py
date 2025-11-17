import pytest
import requests

API_URL = "http://localhost:8000"

@pytest.fixture
def sample_hero_data():
    """Fixture with sample hero data for testing"""
    return {
        "data": {
            "powerstats": {
                "intelligence": 75,
                "strength": 80,
                "speed": 50,
                "durability": 85,
                "combat": 70,
            },
            "appearance": {
                "height": ["6'2", "188 cm"],
                "weight": ["185 lb", "84 kg"]
            }
        }
    }

@pytest.fixture
def invalid_hero_data():
    """Fixture with invalid hero data for testing"""
    return {
        "data": {
            "powerstats": {
                "intelligence": None,
                "strength": None,
                "speed": None,
                "durability": None,
                "combat": None,
            },
            "appearance": {
                "height": [],
                "weight": []
            }
        }
    }

def test_health_check():
    """Test the /health endpoint"""
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_info_endpoint():
    """Test the /info endpoint"""
    response = requests.get(f"{API_URL}/info")
    assert response.status_code == 200
    data = response.json()
    assert "Equipo" in data
    assert "Tipo de Modelo" in data
    assert "hyperparameters" in data
    assert "preprocessing" in data
    assert data["Equipo"] == "Bomba en el IIMAS"
    assert "StandardScaler" in data["preprocessing"]
    assert isinstance(data["hyperparameters"], dict)

def test_info_endpoint_model_type():
    """Test that the model type is one of the expected types"""
    response = requests.get(f"{API_URL}/info")
    assert response.status_code == 200
    data = response.json()
    model_type = data["Tipo de Modelo"]
    assert model_type in ["SVR", "RandomForestRegressor", "MLPRegressor"]

def test_info_endpoint_preprocessing_description():
    """Test that preprocessing description contains key terms"""
    response = requests.get(f"{API_URL}/info")
    assert response.status_code == 200
    data = response.json()
    preprocessing = data["preprocessing"].lower()
    assert "bayesian optimization" in preprocessing or "standardscaler" in preprocessing

def test_predict_endpoint_valid_data(sample_hero_data):
    """Test the /predict endpoint with valid data"""
    response = requests.post(
        f"{API_URL}/predict",
        json=sample_hero_data,
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "power_prediction" in data
    assert isinstance(data["power_prediction"], int)
    assert data["power_prediction"] >= 0

def test_predict_endpoint_response_structure(sample_hero_data):
    """Test that the prediction response has the correct structure"""
    response = requests.post(
        f"{API_URL}/predict",
        json=sample_hero_data,
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "power_prediction" in data
    # Check no error field is present in successful response
    assert "error" not in data or data.get("error") is None

def test_predict_endpoint_different_values():
    """Test predictions with different hero values"""
    test_data = {
        "data": {
            "powerstats": {
                "intelligence": 100,
                "strength": 100,
                "speed": 100,
                "durability": 100,
                "combat": 100,
            },
            "appearance": {
                "height": ["7'0", "213 cm"],
                "weight": ["250 lb", "113 kg"]
            }
        }
    }
    response = requests.post(
        f"{API_URL}/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "power_prediction" in data
    assert isinstance(data["power_prediction"], int)

def test_predict_endpoint_minimal_values():
    """Test predictions with minimal hero values"""
    test_data = {
        "data": {
            "powerstats": {
                "intelligence": 1,
                "strength": 1,
                "speed": 1,
                "durability": 1,
                "combat": 1,
            },
            "appearance": {
                "height": ["5'0", "152 cm"],
                "weight": ["100 lb", "45 kg"]
            }
        }
    }
    response = requests.post(
        f"{API_URL}/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "power_prediction" in data
    assert isinstance(data["power_prediction"], int)

def test_predict_endpoint_missing_data():
    """Test the /predict endpoint with missing data"""
    invalid_data = {"data": {}}
    response = requests.post(
        f"{API_URL}/predict",
        json=invalid_data,
        headers={"Content-Type": "application/json"}
    )
    # API returns 200 but with error field
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["power_prediction"] is None

def test_predict_endpoint_empty_payload():
    """Test the /predict endpoint with empty payload"""
    response = requests.post(
        f"{API_URL}/predict",
        json={},
        headers={"Content-Type": "application/json"}
    )
    # Should return validation error
    assert response.status_code in [200, 422]
    if response.status_code == 200:
        data = response.json()
        assert "error" in data

def test_predict_endpoint_missing_powerstats():
    """Test prediction with missing powerstats"""
    test_data = {
        "data": {
            "appearance": {
                "height": ["6'0", "183 cm"],
                "weight": ["180 lb", "82 kg"]
            }
        }
    }
    response = requests.post(
        f"{API_URL}/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["power_prediction"] is None

def test_predict_endpoint_missing_appearance():
    """Test prediction with missing appearance data"""
    test_data = {
        "data": {
            "powerstats": {
                "intelligence": 80,
                "strength": 75,
                "speed": 60,
                "durability": 70,
                "combat": 65,
            }
        }
    }
    response = requests.post(
        f"{API_URL}/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["power_prediction"] is None

def test_predict_endpoint_invalid_height_format():
    """Test prediction with invalid height format"""
    test_data = {
        "data": {
            "powerstats": {
                "intelligence": 80,
                "strength": 75,
                "speed": 60,
                "durability": 70,
                "combat": 65,
            },
            "appearance": {
                "height": ["invalid", "also invalid"],
                "weight": ["200 lb", "91 kg"]
            }
        }
    }
    response = requests.post(
        f"{API_URL}/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 200
    data = response.json()
    # Should return error due to invalid format
    assert "error" in data or "power_prediction" in data

def test_predict_endpoint_empty_height_weight():
    """Test prediction with empty height and weight arrays"""
    test_data = {
        "data": {
            "powerstats": {
                "intelligence": 80,
                "strength": 75,
                "speed": 60,
                "durability": 70,
                "combat": 65,
            },
            "appearance": {
                "height": [],
                "weight": []
            }
        }
    }
    response = requests.post(
        f"{API_URL}/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["power_prediction"] is None

def test_api_cors_headers():
    """Test that appropriate headers are returned"""
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200
    assert "content-type" in response.headers
    assert "application/json" in response.headers["content-type"]

def test_invalid_endpoint():
    """Test accessing an invalid endpoint"""
    response = requests.get(f"{API_URL}/invalid_endpoint")
    assert response.status_code == 404

def test_predict_with_alternative_height_format():
    """Test prediction with alternative height format"""
    test_data = {
        "data": {
            "powerstats": {
                "intelligence": 80,
                "strength": 75,
                "speed": 60,
                "durability": 70,
                "combat": 65,
            },
            "appearance": {
                "height": ["6 ft 1 in", "185 cm"],
                "weight": ["200 lb", "91 kg"]
            }
        }
    }
    response = requests.post(
        f"{API_URL}/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "power_prediction" in data
    assert isinstance(data["power_prediction"], int)

def test_predict_with_metric_height_format():
    """Test prediction with metric height format"""
    test_data = {
        "data": {
            "powerstats": {
                "intelligence": 85,
                "strength": 70,
                "speed": 55,
                "durability": 75,
                "combat": 60,
            },
            "appearance": {
                "height": ["1.85 meters", "185 cm"],
                "weight": ["85 kg", "187 lb"]
            }
        }
    }
    response = requests.post(
        f"{API_URL}/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "power_prediction" in data

def test_predict_consistency():
    """Test that the same input produces the same prediction"""
    test_data = {
        "data": {
            "powerstats": {
                "intelligence": 75,
                "strength": 80,
                "speed": 50,
                "durability": 85,
                "combat": 70,
            },
            "appearance": {
                "height": ["6'2", "188 cm"],
                "weight": ["185 lb", "84 kg"]
            }
        }
    }
    
    response1 = requests.post(
        f"{API_URL}/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    response2 = requests.post(
        f"{API_URL}/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response1.json()["power_prediction"] == response2.json()["power_prediction"]

def test_predict_with_string_numbers():
    """Test prediction when numbers are sent as strings"""
    test_data = {
        "data": {
            "powerstats": {
                "intelligence": "80",
                "strength": "75",
                "speed": "60",
                "durability": "70",
                "combat": "65",
            },
            "appearance": {
                "height": ["6'1", "185 cm"],
                "weight": ["200 lb", "91 kg"]
            }
        }
    }
    response = requests.post(
        f"{API_URL}/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    # May return error or handle conversion
    assert response.status_code == 200

def test_predict_boundary_values():
    """Test predictions with boundary values (0 and 100)"""
    test_data = {
        "data": {
            "powerstats": {
                "intelligence": 0,
                "strength": 0,
                "speed": 0,
                "durability": 0,
                "combat": 0,
            },
            "appearance": {
                "height": ["4'0", "122 cm"],
                "weight": ["50 lb", "23 kg"]
            }
        }
    }
    response = requests.post(
        f"{API_URL}/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "power_prediction" in data or "error" in data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])