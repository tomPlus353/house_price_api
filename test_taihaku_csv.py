"""Run API tests against the Taihaku Ward MLIT sales CSV.

This script converts MLIT transaction rows into the feature schema expected by
the prediction API, submits each row to the endpoint, and compares the
prediction with the actual transaction price in the CSV.

Usage:
    python3 test_taihaku_csv.py
    python3 test_taihaku_csv.py --limit 25
    python3 test_taihaku_csv.py --base-url http://localhost:8000/predict
    python3 test_taihaku_csv.py --price-info 成約価格情報
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple


try:
    import requests
except Exception:
    print("The 'requests' package is required. Install with: pip install requests")
    raise

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "api"))

try:
    from utils.validator import load_allowed_categories, validate_categorical_inputs
except Exception as exc:
    print(
        "The API validator dependencies are required. "
        "Install project requirements before running this script."
    )
    raise


DEFAULT_BASE_URL = "https://tomplus353--ml-model-api-predict.modal.run"
# DEFAULT_CSV_PATH = "Miyagi Prefecture_Taihaku Ward_20244_20253.csv" # 2024-2025, after training period, to test generalization
# DEFAULT_CSV_PATH = "Miyagi Prefecture_Taihaku Ward_20174_20184.csv" #2017-2018, within data training period
DEFAULT_CSV_PATH = "Miyagi Prefecture_Taihaku Ward_20104_20114.csv"  # 2010-2011, within data training period
DEFAULT_OUTPUT_PATH = "taihaku_ward_api_test_results.csv"
DEFAULT_MODEL_DIR = os.path.join("api", "model_renovation")

MUNICIPALITY_MAP = {
    "仙台市太白区": "Taihaku Ward,Sendai City",
}

CITY_PLANNING_MAP = {
    "１低住専": "Category I Exclusively Low-story Residential Zone",
    "２低住専": "Category II Exclusively Low-story Residential Zone",
    "１中住専": "Category I Exclusively Medium-high Residential Zone",
    "２中住専": "Category II Exclusively Medium-high Residential Zone",
    "１種住居": "Category I Residential Zone",
    "２種住居": "Category II Residential Zone",
    "近隣商業": "Neighborhood Commercial Zone",
    "商業": "Commercial Zone",
    "準工業": "Quasi-industrial Zone",
    "工業": "Industrial Zone",
    "調整区域": "Urbanization Control Area",
}

STRUCTURE_MAP = {
    "木造": "W",
    "軽量鉄骨造": "LS",
    "鉄骨造": "S",
    "ＲＣ": "RC",
}

STATION_MAP = {
    "仙台": "Sendai",
    "八木山動物公園": "Yagiyama Zoological Park",
    "南仙台": "Minamisendai",
    "太子堂": "Taishido",
    "富沢": "Tomizawa",
    "愛子": "Ayashi",
    "愛宕橋": "Atagobashi",
    "河原町(宮城)": "Kawaramachi (Miyagi)",
    "陸前白沢": "Rikuzenshirasawa",
    "長町": "Nagamachi",
    "長町一丁目": "Nagamachi 1-chome",
    "長町南": "Nagamachiminami",
}

DISTRICT_MAP = {
    "ひより台": "Hiyoridai",
    "三神峯": "Mikamine",
    "上野山": "Kaminoyama",
    "中田": "Nakada",
    "中田町": "Nakadamachi",
    "二ツ沢": "Futatsusawa",
    "人来田": "Hitokita",
    "八木山南": "Yagiyamaminami",
    "八木山弥生町": "Yagiyamayayoicho",
    "八木山本町": "Yagiyamahoncho",
    "八木山東": "Yagiyamahigashi",
    "八木山緑町": "Yagiyamamidoricho",
    "八木山香澄町": "Yagiyamakasumicho",
    "八本松": "Hachihommatsu",
    "向山": "Mukaiyama",
    "四郎丸": "Shiromaru",
    "佐保山": "Sahoyama",
    "砂押南町": "Sunaoshiminamimachi",
    "土手内": "Doteuchi",
    "大塒町": "Otoyamachi",
    "大谷地": "Oyachi",
    "大野田": "Onoda",
    "太子堂": "Taishido",
    "太白": "Taihaku",
    "富沢": "Tomizawa",
    "富沢南": "Tomizawaminami",
    "富沢西": "Tomizawa",
    "富田": "Tomita",
    "山田上ノ台町": "Yamadauenodaicho",
    "山田本町": "Yamadahoncho",
    "山田自由ケ丘": "Yamadajiyuugaoka",
    "恵和町": "Keiwamachi",
    "日本平": "Nihondaira",
    "東中田": "Higashinakada",
    "東大野田": "Higashionoda",
    "東郡山": "Higashikooriyama",
    "松が丘": "Matsugaoka",
    "柳生": "Yanagiu",
    "根岸町": "Negishimachi",
    "桜木町": "Sakuragimachi",
    "泉崎": "Izumizaki",
    "砂押町": "Sunaochimachi",
    "諏訪町": "Suwamachi",
    "秋保町長袋": "Akiumachinagafukuro",
    "秋保町湯元": "Akiumachiyumoto",
    "秋保町湯向": "Akiumachiyumukai",
    "緑ケ丘": "Midorigaoka",
    "羽黒台": "Hagurodai",
    "芦の口": "Ashinokuchi",
    "若葉町": "Wakabamachi",
    "茂庭": "Moniwa",
    "茂庭台": "Moniwadai",
    "萩ケ丘": "Hagigaoka",
    "袋原": "Fukurobara",
    "八木山松波町": "Yagiyamamatsunamicho",
    "西の平": "Nishinodaira",
    "西中田": "Nishinakada",
    "西多賀": "Nishitaga",
    "越路": "Koeji",
    "郡山": "Kooriyama",
    "金剛沢": "Kongozawa",
    "鈎取": "Kagitori",
    "鈎取本町": "Kagitorihoncho",
    "長嶺": "Nagamine",
    "長町": "Nagamachi",
    "長町南": "Nagamachiminami",
    "門前町": "Monzenmachi",
    "青山": "Aoyama",
    "鹿野": "Kano",
    "鹿野本町": "Shikanohoncho",
}


def parse_prediction(pred_value: Any) -> float:
    """Parse API prediction values returned as numbers or yen-formatted strings."""
    if isinstance(pred_value, (int, float)):
        return float(pred_value)

    if not isinstance(pred_value, str):
        raise ValueError(f"Cannot parse prediction value: {pred_value!r}")

    value = pred_value.strip()
    if value.endswith("円"):
        value = value[:-1]
    value = value.replace(",", "")
    return float(value)


def parse_float(value: str) -> Optional[float]:
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped:
        return None

    normalized = stripped.replace(",", "")
    try:
        return float(normalized)
    except ValueError:
        pass

    # MLIT CSVs sometimes use bucketed values like `2,000㎡以上` or `50.0m以上`.
    # For testing, use the numeric lower bound instead of failing the whole row.
    match = re.search(r"\d+(?:\.\d+)?", normalized)
    if match:
        return float(match.group(0))

    return None


def parse_int(value: str) -> Optional[int]:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return int(parsed)


def parse_year(value: str) -> Optional[float]:
    if not value:
        return None
    return float(value.replace("年", "").strip())


def parse_sale_year(value: str) -> Optional[int]:
    if not value:
        return None
    return int(value.split("年", 1)[0])


def parse_station_minutes(value: str) -> Tuple[Optional[float], Optional[float]]:
    if not value:
        return None, None

    direct_mappings = {
        "30分～60分": (30.0, 60.0),
        "1H～1H30": (60.0, 90.0),
        "1H30～2H": (90.0, 120.0),
        "2H～": (120.0, 120.0),
    }
    if value in direct_mappings:
        return direct_mappings[value]

    minutes = parse_float(value)
    if minutes is not None:
        return minutes, minutes

    return None, None


def build_station_time_lookup(rows: Iterable[Dict[str, str]]) -> Dict[str, Tuple[float, float]]:
    """Build same-file median station times from rows with observed station-time values."""
    station_pairs: Dict[str, List[Tuple[float, float]]] = {}

    for row in rows:
        station_jp = row.get("最寄駅：名称", "")
        if not station_jp:
            continue

        station = STATION_MAP.get(station_jp)
        if not station:
            continue

        min_station, max_station = parse_station_minutes(row.get("最寄駅：距離（分）", ""))
        if min_station is None or max_station is None:
            continue

        station_pairs.setdefault(station, []).append((min_station, max_station))

    station_lookup: Dict[str, Tuple[float, float]] = {}
    for station, pairs in station_pairs.items():
        min_values = [pair[0] for pair in pairs]
        max_values = [pair[1] for pair in pairs]
        station_lookup[station] = (
            statistics.median(min_values),
            statistics.median(max_values),
        )

    return station_lookup


def classify_mapped_value(
    field_name: str,
    raw_value: str,
    mapped_value: Any,
) -> Optional[str]:
    if mapped_value is not None:
        return None
    if raw_value:
        return f"unmapped_{field_name}"
    return f"missing_{field_name}"


def classify_required_value(field_name: str, value: Any) -> Optional[str]:
    if value is not None:
        return None
    return f"missing_{field_name}"


def format_reason_details(details: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in sorted(details):
        parts.append(f"{key}={details[key]!r}")
    return "; ".join(parts)


def build_payload(
    row: Dict[str, str],
    station_time_lookup: Dict[str, Tuple[float, float]],
) -> Tuple[Optional[Dict[str, List[Any]]], List[str], Dict[str, Any]]:
    """Convert a CSV row into the API payload format."""
    reasons: List[str] = []
    metadata: Dict[str, Any] = {}
    failure_details: Dict[str, Any] = {}

    municipality_raw = row["市区町村名"]
    district_raw = row["地区名"]
    station_raw = row["最寄駅：名称"]
    city_planning_raw = row["都市計画"]
    structure_raw = row["建物の構造"]

    municipality = MUNICIPALITY_MAP.get(municipality_raw)
    district_name = DISTRICT_MAP.get(district_raw)
    station = STATION_MAP.get(station_raw) if station_raw else None
    city_planning = CITY_PLANNING_MAP.get(city_planning_raw)
    structure = STRUCTURE_MAP.get(structure_raw) if structure_raw else None

    area = parse_float(row["面積（㎡）"])
    total_floor_area = parse_float(row["延床面積（㎡）"])
    building_year = parse_year(row["建築年"])
    coverage_ratio = parse_float(row["建ぺい率（％）"])
    floor_area_ratio = parse_float(row["容積率（％）"])
    year = parse_sale_year(row["取引時期"])
    min_station, max_station = parse_station_minutes(row["最寄駅：距離（分）"])

    mapped_required_fields = [
        ("Municipality", municipality_raw, municipality),
        ("DistrictName", district_raw, district_name),
        ("NearestStation", station_raw, station),
        ("CityPlanning", city_planning_raw, city_planning),
        ("Structure", structure_raw, structure),
    ]
    plain_required_fields = [
        ("Area", area),
        ("TotalFloorArea", total_floor_area),
        ("BuildingYear", building_year),
        ("CoverageRatio", coverage_ratio),
        ("FloorAreaRatio", floor_area_ratio),
        ("Year", year),
    ]

    for field_name, raw_value, mapped_value in mapped_required_fields:
        reason = classify_mapped_value(field_name, raw_value, mapped_value)
        if reason is not None:
            reasons.append(reason)
            failure_details[field_name] = raw_value

    for field_name, field_value in plain_required_fields:
        reason = classify_required_value(field_name, field_value)
        if reason is not None:
            reasons.append(reason)
            failure_details[field_name] = row.get(
                {
                    "Area": "面積（㎡）",
                    "TotalFloorArea": "延床面積（㎡）",
                    "BuildingYear": "建築年",
                    "CoverageRatio": "建ぺい率（％）",
                    "FloorAreaRatio": "容積率（％）",
                    "Year": "取引時期",
                }[field_name],
                "",
            )

    if min_station is None or max_station is None:
        if station is None:
            if station_raw:
                reasons.append("unmapped_station_time")
                failure_details["station_time"] = row["最寄駅：距離（分）"]
            else:
                reasons.append("missing_station_time")
                failure_details["station_time"] = row["最寄駅：距離（分）"]
        else:
            imputed_times = station_time_lookup.get(station)
            if imputed_times is None:
                reasons.append("missing_station_time")
                failure_details["station_time"] = row["最寄駅：距離（分）"]
            else:
                min_station, max_station = imputed_times
                metadata["station_time_source"] = "imputed_station_median"

    if "station_time_source" not in metadata and min_station is not None and max_station is not None:
        metadata["station_time_source"] = "observed"

    if failure_details:
        metadata["failure_details"] = failure_details

    if reasons:
        return None, reasons, metadata

    payload = {
        "Municipality": [municipality],
        "DistrictName": [district_name],
        "Area": [area],
        "TotalFloorArea": [total_floor_area],
        "BuildingYear": [building_year],
        "CoverageRatio": [coverage_ratio],
        "FloorAreaRatio": [floor_area_ratio],
        "MaxTimeToNearestStation": [max_station],
        "MinTimeToNearestStation": [min_station],
        "NearestStation": [station],
        "CityPlanning": [city_planning],
        "Structure": [structure],
        "Year": [year],
    }
    metadata["station_time_min_used"] = min_station
    metadata["station_time_max_used"] = max_station
    return payload, [], metadata


def validate_payload(
    payload: Dict[str, List[Any]],
    allowed_mapping: Dict[str, List[str]],
) -> Tuple[Dict[str, List[Any]], List[str], Dict[str, Any]]:
    """Validate categorical payload values against the API validator."""
    validated_payload, validation_errors = validate_categorical_inputs(
        payload, allowed_mapping
    )
    metadata: Dict[str, Any] = {}
    reasons: List[str] = []
    failure_details: Dict[str, Any] = {}

    if validation_errors:
        metadata["validation_errors"] = validation_errors
        for field_name in sorted(validation_errors):
            reasons.append(f"unmapped_{field_name}")
            issue = validation_errors[field_name][0]
            failure_details[field_name] = issue.get("value")
    if failure_details:
        metadata["failure_details"] = failure_details

    return validated_payload, reasons, metadata


def row_name(row: Dict[str, str], row_number: int) -> str:
    district = row.get("地区名") or "UnknownDistrict"
    period = row.get("取引時期") or "UnknownPeriod"
    return f"row_{row_number}_{district}_{period}"


def call_api(payload: Dict[str, List[Any]], base_url: str) -> Dict[str, Any]:
    response = requests.post(base_url, json={"data": payload}, timeout=20)
    response.raise_for_status()
    body = response.json()

    if "prediction" not in body:
        raise ValueError(f"No 'prediction' in response: {body}")

    return body


def iter_rows(csv_path: str) -> Iterable[Dict[str, str]]:
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def run_tests(
    csv_path: str,
    base_url: str,
    output_path: str,
    limit: Optional[int],
    price_info: Optional[str],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    filtered_rows: List[Tuple[int, Dict[str, str]]] = []
    allowed_mapping = load_allowed_categories(DEFAULT_MODEL_DIR)

    for row_number, row in enumerate(iter_rows(csv_path), start=2):
        if price_info and row.get("価格情報区分") != price_info:
            continue
        filtered_rows.append((row_number, row))

    station_time_lookup = build_station_time_lookup(row for _, row in filtered_rows)

    for row_number, row in filtered_rows:
        if limit is not None and len(results) >= limit:
            break

        result: Dict[str, Any] = {
            "row_number": row_number,
            "name": row_name(row, row_number),
            "price_info": row.get("価格情報区分", ""),
            "district_jp": row.get("地区名", ""),
            "district_en": DISTRICT_MAP.get(row.get("地区名", "")),
            "station_jp": row.get("最寄駅：名称", ""),
            "station_en": STATION_MAP.get(row.get("最寄駅：名称", "")),
            "trade_period": row.get("取引時期", ""),
            "actual_price_yen": parse_int(row.get("取引価格（総額）", "")),
        }

        payload, reasons, metadata = build_payload(row, station_time_lookup)
        result.update(metadata)
        if payload is None:
            result["status"] = "skipped"
            result["skip_reason"] = "|".join(sorted(set(reasons)))
            if result.get("failure_details"):
                result["skip_detail"] = format_reason_details(result["failure_details"])
            print(
                f"{result['name']}: SKIPPED - {result['skip_reason']}"
                + (
                    f" | {result['skip_detail']}"
                    if result.get("skip_detail")
                    else ""
                )
            )
            results.append(result)
            continue

        validated_payload, validation_reasons, validation_metadata = validate_payload(
            payload, allowed_mapping
        )
        result.update(validation_metadata)
        if validation_reasons:
            result["status"] = "skipped"
            result["skip_reason"] = "|".join(sorted(set(validation_reasons)))
            if result.get("failure_details"):
                result["skip_detail"] = format_reason_details(result["failure_details"])
            print(
                f"{result['name']}: SKIPPED - {result['skip_reason']}"
                + (
                    f" | {result['skip_detail']}"
                    if result.get("skip_detail")
                    else ""
                )
            )
            results.append(result)
            continue

        try:
            api_response = call_api(validated_payload, base_url)
            predicted_yen = parse_prediction(api_response["prediction"])
        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
            results.append(result)
            continue

        actual_yen = result["actual_price_yen"]
        diff_yen = predicted_yen - actual_yen if actual_yen is not None else None
        abs_diff_yen = abs(diff_yen) if diff_yen is not None else None
        ape_pct = (
            abs_diff_yen / actual_yen * 100
            if actual_yen not in (None, 0) and abs_diff_yen is not None
            else None
        )

        result.update(
            {
                "status": "ok",
                "predicted_yen": predicted_yen,
                "diff_yen": diff_yen,
                "abs_diff_yen": abs_diff_yen,
                "ape_pct": ape_pct,
            }
        )
        results.append(result)

        print(
            f"{result['name']}: Predicted {predicted_yen:,.0f}円 | "
            f"Actual {actual_yen:,.0f}円 | "
            f"Abs diff {abs_diff_yen:,.0f}円 | "
            f"APE {ape_pct:.2f}%"
        )

    write_results_csv(results, output_path)
    print_summary(results, output_path)
    return results


def write_results_csv(results: List[Dict[str, Any]], output_path: str) -> None:
    fieldnames = [
        "row_number",
        "name",
        "status",
        "price_info",
        "district_jp",
        "district_en",
        "station_jp",
        "station_en",
        "trade_period",
        "actual_price_yen",
        "station_time_source",
        "station_time_min_used",
        "station_time_max_used",
        "predicted_yen",
        "diff_yen",
        "abs_diff_yen",
        "ape_pct",
        "skip_reason",
        "skip_detail",
        "failure_details",
        "validation_errors",
        "error",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in fieldnames})


def print_summary(results: List[Dict[str, Any]], output_path: str) -> None:
    ok_results = [result for result in results if result.get("status") == "ok"]
    skipped_results = [
        result for result in results if result.get("status") == "skipped"
    ]
    error_results = [result for result in results if result.get("status") == "error"]

    print("\nSummary")
    print(f"Results written to: {output_path}")
    print(f"Rows processed: {len(results)}")
    print(f"Successful tests: {len(ok_results)}")
    print(f"Skipped rows: {len(skipped_results)}")
    print(f"Request errors: {len(error_results)}")

    if ok_results:
        abs_diffs = [result["abs_diff_yen"] for result in ok_results]
        ape_values = [
            result["ape_pct"]
            for result in ok_results
            if result.get("ape_pct") is not None
        ]
        station_time_counts: Dict[str, int] = {}
        for result in ok_results:
            source = result.get("station_time_source", "unknown")
            station_time_counts[source] = station_time_counts.get(source, 0) + 1
        print(f"MAE: {statistics.mean(abs_diffs):,.0f}円")
        print(f"Median abs error: {statistics.median(abs_diffs):,.0f}円")
        if ape_values:
            print(f"MAPE: {statistics.mean(ape_values):.2f}%")
            print(f"Median APE: {statistics.median(ape_values):.2f}%")
        print("Station time sources:")
        for source, count in sorted(
            station_time_counts.items(), key=lambda item: (-item[1], item[0])
        ):
            print(f"- {source}: {count}")

    if skipped_results:
        counts: Dict[str, int] = {}
        for result in skipped_results:
            reason = result.get("skip_reason", "unknown")
            counts[reason] = counts.get(reason, 0) + 1
        print("Skip reasons:")
        for reason, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        ):
            print(f"- {reason}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run API tests using Taihaku Ward MLIT real-estate sales data."
    )
    parser.add_argument(
        "--csv", default=DEFAULT_CSV_PATH, help="Path to the input CSV file."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("APP_URL", DEFAULT_BASE_URL),
        help="Prediction endpoint URL.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write the per-row results CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows to process after filtering.",
    )
    parser.add_argument(
        "--price-info",
        default=None,
        choices=["不動産取引価格情報", "成約価格情報"],
        help="Optional filter for a single MLIT price-information type.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_tests(
        csv_path=args.csv,
        base_url=args.base_url,
        output_path=args.output,
        limit=args.limit,
        price_info=args.price_info,
    )


if __name__ == "__main__":
    main()
