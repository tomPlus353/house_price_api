"""Simple API test for the house price prediction endpoint.

Sends a POST to `http://localhost:8000/predict` with the provided
`test_input` and compares the returned prediction to the expected
real asking price (1,990万円 -> 19,900,000円). The script prints the
predicted value, expected value, absolute difference and percent diff.

Run:
    python3 tests.py

If `requests` is not installed, install it with:
    pip install requests
"""

import json
from typing import Dict, Any, List

try:
    import requests
except Exception:
    print("The 'requests' package is required. Install with: pip install requests")
    raise


def parse_prediction(pred_str: str) -> float:
    """Parse strings like '19,900,000.00円' into float yen value."""
    s = pred_str.strip()
    if s.endswith("円"):
        s = s[:-1]
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError as exc:
        raise ValueError(f"Cannot parse prediction value: {pred_str}") from exc


TEST_CASES: List[Dict[str, Any]] = [
    {
        "name": "Miyagino Odawara - 113.56 m2",
        "input": {
            "Municipality": ["Miyagino Ward,Sendai City"],
            "DistrictName": ["Odawara"],
            "Area": [113.56],
            "TotalFloorArea": [103.27],
            "BuildingYear": [2025.66],
            "CoverageRatio": [60.0],
            "FloorAreaRatio": [200.0],
            "MaxTimeToNearestStation": [17.0],
            "MinTimeToNearestStation": [23.0],
            "NearestStation": ["Tsutsujigaoka"],
            "CityPlanning": ["Category II Residential Zone"],
            "Structure": ["W"],
            "Year": [2025],
        },
        "expected_man_yen": 4780,
    },
    {
        "name": "Miyagino Odawara - 78.61 m2",
        "input": {
            "Municipality": ["Miyagino Ward,Sendai City"],
            "DistrictName": ["Odawara"],
            "Area": [78.61],
            "TotalFloorArea": [84.04],
            "BuildingYear": [2024.33],
            "CoverageRatio": [60.0],
            "FloorAreaRatio": [160.0],
            "MaxTimeToNearestStation": [29.0],
            "MinTimeToNearestStation": [20.0],
            "NearestStation": ["Tsutsujigaoka"],
            "CityPlanning": ["Category II Residential Zone"],
            "Structure": ["W"],
            "Year": [2025],
        },
        "expected_man_yen": 3880,
    },
    {
        "name": "Minaminakayama - Izumichuo",
        "input": {
            "Municipality": ["Izumi Ward,Sendai City"],
            "DistrictName": ["Minaminakayama"],
            "Area": [225.09],
            "TotalFloorArea": [129.89],
            "BuildingYear": [1996.0],
            "CoverageRatio": [40.0],
            "FloorAreaRatio": [60.0],
            "MaxTimeToNearestStation": [90.0],
            "MinTimeToNearestStation": [60.0],
            "NearestStation": ["Izumichuo"],
            "CityPlanning": ["Category I Exclusively Low-story Residential Zone"],
            "Structure": ["W"],
            "Year": [2025],
        },
        "expected_man_yen": 1990,
    },
    {
        "name": "Asahigaoka - Kuromatsu",
        "input": {
            "Municipality": ["Izumi Ward,Sendai City"],
            "DistrictName": ["Asahigaoka"],
            "Area": [100.19],
            "TotalFloorArea": [34.83],
            "BuildingYear": [1968.0],
            "CoverageRatio": [50.0],
            "FloorAreaRatio": [80.0],
            "MaxTimeToNearestStation": [20.0],
            "MinTimeToNearestStation": [5.0],
            "NearestStation": ["Kuromatsu (Miyagi)"],
            "CityPlanning": ["Category I Exclusively Low-story Residential Zone"],
            "Structure": ["W"],
            "Year": [2025],
        },
        "expected_man_yen": 1300,
    },
    {
        "name": "Yamanotera",
        "input": {
            "Municipality": ["Izumi Ward,Sendai City"],
            "DistrictName": ["Yamanotera"],
            "Area": [380.18],
            "TotalFloorArea": [180.94],
            "BuildingYear": [1980.66],
            "CoverageRatio": [50.0],
            "FloorAreaRatio": [80.0],
            "MaxTimeToNearestStation": [28.0],
            "MinTimeToNearestStation": [28.0],
            "NearestStation": ["Izumichuo"],
            "CityPlanning": ["Category I Exclusively Low-story Residential Zone"],
            "Structure": ["W"],
            "Year": [2025],
        },
        "expected_man_yen": 1300,
    },
]


def run_single_test(
    case: Dict[str, Any], base_url: str = "http://localhost:8000/predict"
) -> Dict[str, Any]:
    """Run a single test case against the API and return structured results."""
    payload = {"data": case["input"]}
    try:
        r = requests.post(base_url, json=payload, timeout=15)
    except requests.exceptions.RequestException as exc:
        return {"name": case.get("name"), "error": str(exc)}

    if r.status_code != 200:
        return {"name": case.get("name"), "error": f"HTTP {r.status_code}: {r.text}"}

    try:
        data = r.json()
    except ValueError:
        return {"name": case.get("name"), "error": "Invalid JSON response"}

    if "prediction" not in data:
        return {
            "name": case.get("name"),
            "error": f"No 'prediction' in response: {data}",
        }

    try:
        pred_value = parse_prediction(data["prediction"])
    except ValueError as exc:
        return {"name": case.get("name"), "error": f"Parse error: {exc}", "raw": data}

    expected_yen = case.get("expected_man_yen", 0) * 10_000
    diff = pred_value - expected_yen
    pct = (diff / expected_yen) * 100 if expected_yen else None

    return {
        "name": case.get("name"),
        "predicted_yen": pred_value,
        "expected_yen": expected_yen,
        "diff_yen": diff,
        "diff_pct": pct,
    }


def run_all_tests():
    results = []
    for case in TEST_CASES:
        res = run_single_test(case)
        results.append(res)
        if "error" in res:
            print(f"{res['name']}: ERROR - {res['error']}")
        else:
            print(
                f"{res['name']}: Predicted {res['predicted_yen']:,.2f}円 | Expected {res['expected_yen']:,.2f}円 | Diff {res['diff_yen']:,.2f}円 ({res['diff_pct']:.2f}%)"
            )

    # Summary
    print("\nSummary")
    for res in results:
        if "error" in res:
            print(f"- {res['name']}: ERROR - {res['error']}")
        else:
            print(
                f"- {res['name']}: diff {res['diff_yen']:,.2f}円 ({res['diff_pct']:.2f}%)"
            )

    return results


if __name__ == "__main__":
    run_all_tests()
