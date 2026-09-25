# References and Provenance Registry

This document records the bibliographic references, technical standards, and provenance classifications utilized in the Materials Data Lab project. All citations are maintained offline without network requests.

---

## Provenance Taxonomy
- **`STANDARD_CROSSCHECKED`**: Verified against recognized technical standards or independent cross-checked literature sources.
- **`REPORT_ONLY`**: Derived from a single literature compilation or dataset card heuristic without independent full-text standard verification.
- **`AMBIGUOUS` / `UNKNOWN`**: Source citation or value cannot be definitively resolved from available metadata; kept explicitly unverified to prevent fabrication.

---

## Dataset Reference
- **Title**: Tempering data for carbon and low alloy steels
- **Publisher / Author**: Raiipa Technologies (Kaggle: `rgerschtzsauer`)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **URL**: [https://www.kaggle.com/datasets/rgerschtzsauer/tempering-data-for-carbon-and-low-alloy-steels](https://www.kaggle.com/datasets/rgerschtzsauer/tempering-data-for-carbon-and-low-alloy-steels)
- **Dataset Version**: Version 3 (file timestamped 2024-05-20; accessed 2026-09-22)
- **Provenance**: `REPORT_ONLY` (dataset documentation card)

---

## Primary Experimental and Metallurgical Literature
1. **Hollomon, J. H., & Jaffe, L. D. (1945)**
   - *Time-temperature relations in tempering steel*. Transactions of the American Institute of Mining and Metallurgical Engineers (AIME), Vol. 162, pp. 223–249.
   - **Provenance**: `STANDARD_CROSSCHECKED`
   - **Notes**: Foundational paper introducing the Hollomon-Jaffe tempering parameter $P = T(c + \log t)$.

2. **Grange, R. A., & Baughman, R. W. (1956)**
   - *Hardness of tempered martensite in carbon and low-alloy steels*. Transactions of the American Society for Metals (ASM), Vol. 48, pp. 165–197.
   - **Provenance**: `STANDARD_CROSSCHECKED`
   - **Notes**: Contains systematic experimental hardness measurements for tempered martensite across numerous AISI-SAE grades. Initial hardness values were not reported in this subset (indicated as literal `?` in dataset).

3. **Kang, S. U., & Lee, Y. K. (2014)**
   - *Prediction of Hardness of Tempered Martensite in Low-Alloy Steels*. Materials Transactions, 55(7), pp. 1069–1072. DOI: 10.2320/matertrans.M2014004.
   - **Provenance**: `STANDARD_CROSSCHECKED`
   - **Notes**: Quantitative formulation of tempering kinetics and hardness evolution in low-alloy steels. (Note: our composite-C is a simplified adaptation, not a reproduction of the original method).

4. **Penha, R. (2010)**
   - **Provenance**: `AMBIGUOUS` (Retained as `UNKNOWN`)
   - **Ambiguity Details**: The Kaggle dataset card references "Penha, 2010" without distinguishing between two candidate academic publications:
     - Candidate (a): *Estudo da cinética de revenimento de aços empregados na indústria automotiva*. Ph.D. Dissertation / Doctoral thesis, Escola de Engenharia de São Carlos, Universidade de São Paulo (EESC-USP), São Carlos, 151 pp., 2010.
     - Candidate (b): *Cinética de revenimento de aços baixo carbono*. Anais do 65º Congresso Anual da ABM (Associação Brasileira de Metalurgia, Materiais e Mineração), Rio de Janeiro, 2010.
   - **Resolution Policy**: Because the dataset author does not specify the exact publication, the provenance cannot be verified and remains designated as `UNKNOWN` / `AMBIGUOUS`.

---

## Standards and Normative References
*(Tam metin erişimi olmaksızın ad bazında referans verilmiştir; `REPORT_ONLY`)*

- **ASTM E18**: *Standard Test Methods for Rockwell Hardness of Metallic Materials*. Establishes the 20–70 HRC valid measurement range.
- **ISO 6508**: *Metallic materials — Rockwell hardness test — Part 1: Test method*. Harmonized international standard for Rockwell C hardness scale.
- **ASTM E140**: *Standard Hardness Conversion Tables for Metals Relationship Among Brinell Hardness, Vickers Hardness, Rockwell Hardness, Superficial Hardness, Knoop Hardness, Scleroscope Hardness, and Leeb Hardness*. Table 1 sets the lower conversion threshold for Rockwell C at 238 HV (~20 HRC). Extrapolations below 20 HRC are flagged as `E140_EXTRAPOLATION_SUSPECT`.
- **SAE J403**: *Chemical Compositions of SAE Carbon Steels*. Defines standard composition limits for AISI-SAE 10xx, 11xx, 12xx, and 15xx series.
- **SAE J404**: *Chemical Compositions of SAE Alloy Steels*. Defines standard composition limits for low-alloy grades (13xx, 40xx, 41xx, 43xx, 51xx, 86xx, etc.).
- **SAE J409**: *Product Analysis Permissible Variations from Specified Chemical Analysis of a Heat or Cast of Steel*.
