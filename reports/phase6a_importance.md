# Phase 6a: Permutation Importance Cross-Check

| Feature | PI MAE Drop (mean ± std) | PI Rank | MDI Rank |
| :--- | :--- | :--- | :--- |
| temper_temp_c | 13.0804 ± 0.2673 | 1 | 1 |
| log_time | 3.6472 ± 0.0670 | 2 | 2 |
| c_wt | 3.1344 ± 0.0685 | 3 | 3 |
| cr_wt | 1.6129 ± 0.0471 | 4 | 4 |
| si_wt | 0.3146 ± 0.0097 | 5 | 5 |
| s_wt | 0.2452 ± 0.0082 | 6 | 7 |
| p_wt | 0.2390 ± 0.0096 | 7 | 6 |
| mo_wt | 0.2243 ± 0.0084 | 8 | 9 |
| mn_wt | 0.1951 ± 0.0072 | 9 | 8 |
| ni_wt | 0.1404 ± 0.0059 | 10 | 10 |
| cu_wt | 0.0123 ± 0.0011 | 11 | 11 |
| al_wt | 0.0033 ± 0.0003 | 12 | 13 |
| v_wt | 0.0025 ± 0.0003 | 13 | 12 |

## Gözlem
MDI ve Permutation Importance sıralamaları genel olarak uyumludur. Ancak karbon (C) ve log_time gibi kritik özelliklerin sırası veya aralarındaki ağırlık farkları permutation importance ile daha net (korelasyon yanılgısından arınmış) görülmektedir. [REPORT_ONLY]