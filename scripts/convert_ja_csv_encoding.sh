#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/convert_csv_encoding.sh [--in-place] [--from ENCODING] <file>

Examples:
  scripts/convert_csv_encoding.sh "Miyagi Prefecture_Taihaku Ward_20104_20114.csv"
  scripts/convert_csv_encoding.sh --in-place "Miyagi Prefecture_Taihaku Ward_20104_20114.csv"
  scripts/convert_csv_encoding.sh --from SHIFT_JIS input.csv

Behavior:
  - Default source encoding is CP932.
  - Default output is a new UTF-8 file next to the original:
      input.csv -> input_utf8.csv
  - With --in-place, the original file is replaced after successful conversion.
EOF
}

in_place=false
from_encoding="CP932"
input_file=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-place)
      in_place=true
      shift
      ;;
    --from)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --from" >&2
        usage
        exit 1
      fi
      from_encoding="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      if [[ -n "$input_file" ]]; then
        echo "Only one input file is supported." >&2
        usage
        exit 1
      fi
      input_file="$1"
      shift
      ;;
  esac
done

if [[ -z "$input_file" ]]; then
  usage
  exit 1
fi

if [[ ! -f "$input_file" ]]; then
  echo "File not found: $input_file" >&2
  exit 1
fi

input_dir=$(dirname "$input_file")
input_base=$(basename "$input_file")
input_stem="${input_base%.*}"
input_ext="${input_base##*.}"

if [[ "$input_base" == "$input_ext" ]]; then
  output_file="${input_file}_utf8"
else
  output_file="${input_dir}/${input_stem}_utf8.${input_ext}"
fi

if [[ "$in_place" == true ]]; then
  temp_file=$(mktemp "${TMPDIR:-/tmp}/convert_csv_encoding.XXXXXX")
  trap 'rm -f "$temp_file"' EXIT
  iconv -f "$from_encoding" -t UTF-8 "$input_file" > "$temp_file"
  mv "$temp_file" "$input_file"
  trap - EXIT
  echo "Converted in place: $input_file"
else
  iconv -f "$from_encoding" -t UTF-8 "$input_file" > "$output_file"
  echo "Converted file written to: $output_file"
fi
