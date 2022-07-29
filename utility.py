import pandas as pd
import numpy as np
import difflib
import re
import io
import ipywidgets as widgets
from ipywidgets import FileUpload, Layout

class StopExecution(Exception):
    def _render_traceback_(self):
        pass
    
def manual_identification(options, unsure, outer="성별 컬럼"):
    '''
    Adapted from bridge (WM) program, utility.py 
    * Functionality similar with minor comment changes 
    * Handles out of bounds error
    '''
    print("*" * 40)
    print("{}에 '{}' 인식 불가능.".format(outer, unsure))
    for i, opt in enumerate(options):
        print("    ({}) {}".format(i, opt))
        
    response = input("위에 옵션 중 '{}' 있음 (y) 없음 (n) 종료 (q): ".format(unsure))
    while response != "y" and response != "n" and response != "q":
        print("잘못 입력. 다시 선택하세요.")
        response = input("위에 옵션 중 '{}' 있음 (y) 없음 (n) 종료 (q): ".format(unsure))
            
    if response == "y":
        choice_idx = input("    '{}' 옵션을 숫자로 선택하세요: ".format(unsure))

        while not choice_idx.isdigit() or int(choice_idx) not in range(len(options)):
            choice_idx = input("    '{}' 옵션을 다시 선택하세요: ".format(unsure))

        return options[int(choice_idx)]
    
    if response == "q":
        raise StopExecution
        
    return ""


def prompt_upload(description):
    uploader = FileUpload(description=description, layout=Layout(width="250px"), multiple=False)
    display(uploader)

    main_display = widgets.Output()

    def on_upload_change(inputs):
        with main_display:
            main_display.clear_output()
            display(list(inputs['new'].keys())[-1])

    uploader.observe(on_upload_change, names='value')
    return uploader

def read_file(item, sheet=1):
    '''
    All excel files must be formatted from A:G
    '''
    uploaded = item.value
    file_name = list(uploaded.keys())[0]
    
    # Convert sheet to 0-indexing since excel sheets are default 1-indexed
    return pd.read_excel(io.BytesIO(uploaded[file_name]["content"]),
                         sheet_name=sheet - 1,
                         usecols="A:G")

def process_values(df_in):
    string_cols = [1, 3]
    df = df_in.copy(deep=True)

    for i in string_cols:
        # Convert whitespace only cells to NaN
        df = df.replace(to_replace=r"^\s+$", value=np.nan, regex=True)

        # Drop NaN
        df = df.dropna(how="any")

        # Convert from object to string type
        df[df.columns[i]] = df[df.columns[i]].astype(str)

        # Remove trailing/leading whitespace
        df[df.columns[i]] = df[df.columns[i]].str.strip()

        # Make lowercase
        df[df.columns[i]] = df[df.columns[i]].str.lower()
        
    return df

def check_mistakes(df):
    '''
    [1, 3] are the (0-indexed) order of string columns in excel
    '''
    for j in [1, 3]:
        unique = pd.Series(pd.unique(df[df.columns[j]]))
        N = len(unique)
        i = 0
        while i < N:
            others = list(unique)
            others.remove(unique[i])
            close = difflib.get_close_matches(unique[i], others, cutoff=0.8)
            if close:
                for c in close:
                    print("오타 확인:\n    (1) '{}' (O)     '{}' (X)\n    (2) '{}' (X)     '{}' (O)\n    (3) 오타 아니에요 ".format(unique[i], c, unique[i], c))
                    confirm = input("Enter: ")
                    while confirm not in ["1", "2", "3"]:
                        confirm = input("다시 (1, 2, 3): ")

                    if confirm == "1":
                        df[df.columns[j]] = df[df.columns[j]].replace(c, unique[i], regex=False)
                        
                        unique = unique.str.replace(c, unique[i], regex=False)
                        unique = pd.Series(pd.unique(unique))
                        N -= 1
                    elif confirm == "2":
                        df[df.columns[j]] = df[df.columns[j]].replace(unique[i], c, regex=False)
                        
                        unique = unique.str.replace(unique[i], c, regex=False)
                        unique = pd.Series(pd.unique(unique))
                        N -= 1
            i += 1

    return df


def change_unit(df, unit=""):
    '''
    Do not pass in anything in unit parameter
    '''
    df = df.copy()
    if "day" in df.columns[5].lower() or unit == "1":
        print("투여기간 단위: Day")
        pass

    elif "week" in df.columns[5].lower() or unit == "2":
        print("투여기간 단위: Week")
        df[df.columns[5]] = df[df.columns[5]] * 7

    elif "month" in df.columns[5].lower() or unit == "3":
        print("투여기간 단위: Month")
        print("   * Note: 1 Month = 30.417 Days")
        df[df.columns[5]] = df[df.columns[5]] * 30.417

    elif "year" in df.columns[5].lower() or unit == "4":
        print("투여기간 단위: Year")
        print("   * Note: 1 Year = 365.26 Days")
        df[df.columns[5]] = df[df.columns[5]] * 365.26
        
    elif unit == "5":
        print("**Custom Unit**")
        factor = input("    In 1 unit of <custom unit> how many days are there?: ")
        
        while not factor.isdigit():
            factor = input("    Please enter a number: ")
        
        df[df.columns[5]] = df[df.columns[5]] * float(factor)

    else:
        print("'{}' 단위:".format(df.columns[5]))
        print("    (1) Day")
        print("    (2) Week")
        print("    (3) Month")
        print("    (4) Year")
        print("    (5) 다른 단위")
        option = input("Enter: ")

        while option not in ["1", "2", "3", "4", "5"]:
            option = input("Enter (1-5): ")
            
        df = change_unit(df, option)
        
    return df

def check_unit(df):
    '''
    
    '''
    df_unit = change_unit(df)
    df_unit = df_unit.rename(columns={df.columns[5] : "튜여기간 (days)"})
    return df_unit

def check_binary(df_in):   
    df = df_in.copy()
    cleaned = [x for x in df[df.columns[1]] if str(x) != 'nan']

    while len(pd.unique(cleaned)) > 2:
        print("Column '{}' is not binary: ".format(df.columns[1]), end="")
        print(pd.unique(cleaned))

        choice = input("\t모드를 숫자로 선택 후 'Enter'키로 이동하세요:\n\t1) 제목 삭제 모드 (모든 매치 제거) \n\t2) 제목 수정 모드 (오타인 경우) \nEnter: ")

        if choice == "1":
            problem_rows = input("\t\t(정확히) 일치하는 행 제거: " )
            df = df.loc[~df[df.columns[1]].str.contains(problem_rows, regex=False)]

            cleaned = [x for x in df[df.columns[1]] if str(x) != 'nan']
            if len(pd.unique(cleaned)) > 2:
                print("Binary 처리 실패. '{}'".format(df.columns[1]))
                print(pd.unique(cleaned))
                df = check_binary(df)

        elif choice == "2":
            remove = input("\t제거할 문자열: ")
            value = input("\t대체할 문자열: ")
            df[df.columns[1]] = df[df.columns[1]].str.replace(remove, value, regex=False)

            cleaned = [x for x in df[df.columns[1]] if str(x) != 'nan']
            if len(pd.unique(cleaned)) > 2:
                print("Binary 처리 실패. '{}'".format(df.columns[1]))
                print(pd.unique(cleaned))
                df = check_binary(df)

        else:
            print("Enter 1 or 2")
            
        cleaned = [x for x in df[df.columns[1]] if str(x) != 'nan']
    
    return df


def process_source_data(df_in):
    df_in = process_values(df_in)
    df_in = check_mistakes(df_in)
    df_in = check_binary(df_in)
    return check_unit(df_in)

def generate_duration(df_in, dp=2, compute_total=True):
    '''
    (1) 노출기간별 (Duration)
    '''
    print("*" *40)
    print("(1) 노출기간별 (Duration)")
    print("*" *40)
    
    category = pd.cut(df_in[df_in.columns[5]],
                      bins=[0, 1, 6*7, 12*7, 26*7, 52*7, np.inf],
                      right=False,
                      labels=["< 1 Day", "≥ 1 Day", "≥ 6 Weeks", "≥ 12 Weeks", "≥ 26 Weeks", "≥ 52 Weeks"])
    
    
    duration = df_in.groupby(category)[df_in.columns[6]].agg(["count", "sum"]).reset_index()
    duration = duration.rename(columns={duration.columns[0] : "Duration of Exposure",
                                        "count" : "Patients",
                                        "sum" : "Patient Year of Exposure (PYE)"})

    duration = duration.round({"Patient Year of Exposure (PYE)" : dp})
    
    
    if compute_total:
        total_row = duration[['Patients', 'Patient Year of Exposure (PYE)']].sum().to_frame().T
        total_row["Duration of Exposure"] = "Total"
        
        duration = pd.concat([duration, total_row], ignore_index=True)
        
        duration["Patients"] = duration["Patients"].astype("int64")
        
    duration = duration.round({"Patient Year of Exposure (PYE)" : dp})
    return duration


def process_missing_gender(gender_df):
    '''
    '''
    # Added switched variable due to later use for count aggregation
    switched = ""
    
    gender = gender_df.copy(deep=True)
    if gender.shape[1] == 1:
        col = gender.columns[0]
        
        print("Note: only '{}' gender in dataset".format(col))

        if col.lower() in ["m", "male"]:
            # Add F column
            gender["F"] = 0
            switched = "F"

        elif col.lower() in ["f", "female"]:
            # Add M column
            gender["M"] = 0
            switched = "M"

        else:
            # Manual identification
            result = manual_identification(["M", "F"], col)

            if result == "M":
                # Add F column
                gender["F"] = 0
                switched = "F"

            elif result == "F":
                # Add M column
                gender["M"] = 0
                switched = "M"

            else:
                print("성별 컬럼에 Male (M)이나 Female (F)은 둘중에 하나 필요합니다.")
                raise StopExecution

    return gender, switched

def generate_age_gender(df_in, dp=2, compute_total=True):
    '''
    (2) 성별, 연령별 
    '''
    print("*" *40)
    print("(2) 성별, 연령별 ")
    print("*" *40)
    # Pivot Female and Male values into headers
    gender = df_in.pivot_table(index=[df_in.columns[0], df_in.columns[2]],
                               columns=df_in.columns[1],
                               values=df_in.columns[6])
    
    # Handle no Female/Male column cases
    gender, switched = process_missing_gender(gender)
    
    # Guaranteed for F to take priority over M
    gender = gender[gender.columns.sort_values(key=lambda x : x.str.lower())]
    
    # Now we can rename
    gender.columns = ["F", "M"]
    
    gender = gender.reset_index()
    
    # Grouping by age
    category = pd.cut(gender[gender.columns[1]],
                      bins=[0, 12, 18, 65, 75, np.inf],
                      right=False, labels=["0 to 11 years", "12 to 17 years", "18 to 64 years", "65 to 74 years", "≥ 75 years"])
    
    
    gender_multi = gender.groupby(category).agg(["count", "sum"]).swaplevel(axis=1)
    gender_multi = gender_multi.drop(columns=[("count", df_in.columns[2]),
                                              ("sum", df_in.columns[2]),
                                              ("count", df_in.columns[0]),
                                              ("sum", df_in.columns[0])])
    
    # Sum aggregation is OK since zero by intiailization
    # Need to prevent count from aggregating
    if switched:
        gender_multi["count", switched] = 0
        
    cols = pd.MultiIndex.from_tuples([('count', 'M'),
                                  ('count', 'F'), 
                                  ('sum', 'M'),
                                  ('sum', 'F')])
    
    gender = pd.DataFrame(gender_multi, columns=cols).reset_index()
    
    
    # Rename columns
    gender = gender.rename(columns={df_in.columns[2] : "Age group - all indications",
                                "count" : "Patients",
                                "sum" : "Person time (person-year)"})
    
    # Decimal places
    gender = gender.round({('Person time (person-year)', 'M') : dp})
    gender = gender.round({('Person time (person-year)', 'F') : dp})

    
    # Add total row
    if compute_total:
        total_row = gender[['Person time (person-year)', 'Patients']].sum().to_frame().T
        total_row["Age group - all indications"] = "Total"
        
        gender = pd.concat([gender, total_row], ignore_index=True)
        
        gender["Patients"] = gender["Patients"].astype("int64")
    
    return gender

def generate_dose(df_in, dp=2, compute_total=True):
    '''
    (3) 투여용량별 
    '''
    print("*" *40)
    print("(3) 투여용량별 ")
    print("*" *40)
    
    category = pd.cut(df_in[df_in.columns[4]],
                      bins=[-1, 2.5, 5, 10, 15, 20, np.inf],
                      right=True,
                      labels=["≤ 2.5 mg", "> 2.5 mg to ≤ 5 mg", "> 5 mg to ≤ 10 mg", "> 10 mg to ≤ 15 mg", "> 15 mg to ≤ 20 mg", "> 20 mg"])
    
    duration = df_in.groupby(category)[df_in.columns[6]].agg(["count", "sum"]).reset_index()
    duration = duration.rename(columns={duration.columns[0] : "Daily Dose of Exposure",
                                        "count" : "Patients",
                                        "sum" : "Patient Year of Exposure (PYE)"})
    
    
    duration = duration.round({"Patient Year of Exposure (PYE)" : dp})
    
    
    if compute_total:
        total_row = duration[['Patients', 'Patient Year of Exposure (PYE)']].sum().to_frame().T
        total_row["Daily Dose of Exposure"] = "Total"
        
        duration = pd.concat([duration, total_row], ignore_index=False)
        
        duration["Patients"] = duration["Patients"].astype("int64")
        
    duration = duration.round({"Patient Year of Exposure (PYE)" : dp})
    return duration

def process_digits(list_in):
    '''
    Removes trailing and leading whitespace from each string element in list
    Converts each element into an int
    Maintains order
    
    * 2.0 : changed from regex to strip() method. No longer need re library.
    '''
    new_list = list()
    for s in list_in:
        new_list.append(int(s.strip()))
    return new_list


def make_other_race(race_in):
    '''
    Make "other" race 
    
    * Added key index error handling
    
    - race : dataframe with race, count, sum columns only; no total row
    '''
    make = input("'Other' 인종 그룹 만들기 (y/n): ")
    
    while make not in ["y", "n"]:
        make = input("다시 선택하세요. 'Other' 인종 그룹 만들기 (y/n): ")
    
    if make == "y":
        race = race_in.copy()
        
        race_col = race.columns[0]
        for i, r in enumerate(race[race_col]):
            print(("    ({}) {}".format(i, r)))
        print(("    ({}) {}".format(len(race), "종료")))

        comma_list = input("'Other'에 해당하는 행을 선택하세요; 숫자를 쉼표로 (,) 구분 (예: 0, 1, 3): ")
        
        idx = process_digits(comma_list.strip().split(","))
        
        if len(race) in idx:
            raise StopExecution

        while max(idx) > len(race) or min(idx) < 0:
            
            comma_list = input("'Other'에 해당하는 행을 선택하세요; 숫자를 쉼표로 (,) 구분 (예: 0, 1, 3): ")
        
            idx = process_digits(comma_list.strip().split(","))

            if len(race) in idx:
                raise StopExecution
            
        print("    선택: {}".format(race[race_col][idx].values))
            

        other_bool = [x in race[race_col][idx].values for x in race[race_col]]
        others = race.loc[other_bool]
        others_row = others[["count", "sum"]].sum().to_frame().T
        others_row[race_col] = "Other"

        non_others = race.loc[~np.array(other_bool)]

        return pd.concat([non_others, others_row], ignore_index=True)
    return race_in

def generate_race(df_in, dp=2, compute_total=True):
    '''
    (4) 인종별 
    '''
    print("*" *40)
    print("(4) 인종별 ")
    print("*" *40)
    
    race = df_in.groupby(by=df_in.columns[3]).agg(["count", "sum"])[df_in.columns[6]].reset_index()
    
    race = make_other_race(race)
    
    total = race[race.columns[1]].sum() # Default total used for percentage computation
    
    if compute_total:
        total_row = race[['count', 'sum']].sum().to_frame().T
        total_row[race.columns[0]] = "Total"
        
        race = pd.concat([race, total_row], ignore_index=True)
        
        # If we already have a computed total row, use it instead
        total = race.loc[race[race.columns[0]] == "Total"]["count"].values[0]


    # Special string formatting for percentages 
    race["(n [%])"] = race["count"].apply(lambda x : x / total * 100)
        
    
    # Process columns
    race = race.round({"(n [%])" : dp})
    
    race["count"] = race["count"].astype('int64').astype(str) + " (" + race["(n [%])"].astype(str) + ")"
    
    race = race.drop(columns=["(n [%])"])
    
    race = race.rename(columns={race.columns[0] : "Race",
                                "count" : "Patients (n [%])",
                                "sum" : "Person Time (subject-year)"})
    
    race = race.round({"Person Time (subject-year)" : dp})
    
    race["Race"] = race["Race"].str.title()
    
    return race


def export(filename, frames):
    '''
    Filename must have .xlsx at end
    '''
    with pd.ExcelWriter(filename) as writer:
        frames[0].to_excel(writer, sheet_name='(1) 노출기간별 (Duration)', index=False)
        frames[1].to_excel(writer, sheet_name='(2) 성별, 연령별 ', index=True) # Multiindex column
        frames[2].to_excel(writer, sheet_name='(3) 투여용량별', index=False)
        frames[3].to_excel(writer, sheet_name='(4) 인종별', index=False)