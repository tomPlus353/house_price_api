# Japan Real Estate Sales Data Sources

## Summary

For actual sold-price data in Japan, the best source is the Ministry of Land, Infrastructure, Transport and Tourism (MLIT) `不動産情報ライブラリ` (`Real Estate Information Library`).

- Main site: <https://www.reinfolib.mlit.go.jp/>
- It covers `取引価格` and `成約価格`, which are actual transaction / contract prices rather than listing prices.

## Best API / Download Source

The best programmatic source is MLIT's official API and CSV download flow inside `不動産情報ライブラリ`.

- API manual: <https://www.reinfolib.mlit.go.jp/help/apiManual>
- Transaction-price API (`XIT001`): <https://www.reinfolib.mlit.go.jp/help/apiManual/xit001/>
- API application page: <https://www.reinfolib.mlit.go.jp/api/request/>
- Web download/manual page: <https://www.reinfolib.mlit.go.jp/realEstatePrices/manual/>

## Why This Is The Best Source

- It is the official MLIT platform.
- It supports actual real-estate transaction data rather than only advertised listing prices.
- It provides an API intended for structured access.
- It also supports CSV downloads from the search/download UI.

## Coverage Notes

Based on MLIT documentation reviewed on March 26, 2026:

- `不動産取引価格情報` is available from `2005 Q3`.
- `成約価格情報` is available from `2021 Q1`.
- The search/download UI supports both transaction-price data and contract-price data.
- The manual states CSV files can be downloaded for the selected conditions.

## Practical Limitations

- API use requires an application and API key.
- MLIT says approval results are sent by email in about `5 business days`.
- MLIT warns against exposing the API key in browser-side code.
- MLIT notes that repeated heavy access may be restricted.

## Recommendation

Use the MLIT API if this project needs repeatable ingestion into a backend pipeline.

Use the CSV download workflow if:

- you want immediate access without waiting for API approval, or
- you only need one-off extracts for validation, backfills, or experiments.

## Secondary Source

Another useful source for actual closed-transaction information is REINS Market Information:

- <https://www.contract.reins.or.jp/>

This is useful contextually, but MLIT `不動産情報ライブラリ` is the better primary source for an engineering workflow because it has official API/application and CSV download paths in one place.

## Sources

- <https://www.reinfolib.mlit.go.jp/>
- <https://www.reinfolib.mlit.go.jp/help/apiManual>
- <https://www.reinfolib.mlit.go.jp/help/apiManual/xit001/>
- <https://www.reinfolib.mlit.go.jp/api/request/>
- <https://www.reinfolib.mlit.go.jp/realEstatePrices/manual/>
- <https://www.contract.reins.or.jp/>
