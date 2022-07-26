**GENERATE EXPOSURE TABLES**

Given source data, generate:

1. 노출기간 (Duration)
2. 성별, 연령별 
3. Dose 
4. 인종별

참고사항: 엑셀 컬럼 제목들은 중요하지 않으니 (예: 'Subject number'를 아무 문자열로 표현 가능 `0-9한글english.,/;:-+(){}[]!@#$%&*~`등), 컬럼들의 의미는 아래와같이 순서대로 매칭 되있어야합니다.

**source data (.xlsx) 제목 순서**

| A              | B   | C   | D    | E      | F        | G                  | H |
|----------------|-----|-----|------|--------|----------|--------------------|---|
| Subject number | Sex | Age | 인종 | Dosage (mg) | Duration (단위$^{1}$) | Person time (year) |  Comment etc.$^{2}$ |

* $^{1}$자동 인식 단위: day(s), week(s), month(s), year(s).
* $^{2}$컬럼 G가 마지막으로 읽히는 컬럼입니다.

**INSTRUCTIONS**
1. [https://mybinder.org/v2/gh/medisafepv/GenerateTables/main](https://mybinder.org/v2/gh/medisafepv/GenerateTables/main)

2. 왼쪽 파일 탐색기 패널에서 `generate_tables.ipynb` 더블클릭