import csv
import json
import pandas as pd
from src.models.zero_shot_LLM import prompt_llm

def parse_taxonomy_json(json_str):
    """
    Parses a single taxonomy JSON string and returns a list of rows.
    Each row is a dict corresponding to one example.
    """
    rows = []
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("❗ JSONDecodeError:", e)
        print("Attempting auto-fix of common issues...")
        
        # Simple fixes:
        fixed_str = json_str
        
        # Remove trailing commas
        fixed_str = re.sub(r",\s*([}\]])", r"\1", fixed_str)
        
        # Remove any triple backticks if accidentally included
        fixed_str = fixed_str.replace("```json", "").replace("```", "")

        
        # Try again
        try:
            data = json.loads(fixed_str)
        except json.JSONDecodeError as e2:
            # Ask LLM to reprompt if struggling
            fix_prompt = """
            Below is text containing error analysis data. 
            Please reformat this *exactly* as valid JSON matching the following schema:

            {
            "topic": string,
            "error_categories": [
                {
                "label": string,
                "percentage": number,
                "examples": [
                    {
                    "excerpt": string,
                    "article_provided": boolean,
                    "article_failure_reason": string
                    }
                ]
                }
            ],
            "new_categories": [
                {
                "label": string,
                "definition": string
                }
            ]
            }

            Only return valid JSON. No explanations, no extra text.

            Here is the text:

            """
            fix_prompt += fixed_str
            json_str = prompt_llm(fix_prompt)
            try:
                data = json.loads(json_str)
            except:    
                print("❗ Still invalid JSON after auto-fix. Skipping this entry.")
                return rows  # Return empty list
    
    try:
        while isinstance(data, str):
            if not data.strip():
                raise json.JSONDecodeError("Empty string cannot be decoded", data, 0)
            data = json.loads(data)
    except json.JSONDecodeError as e:
        print("❗ JSONDecodeError while peeling nested strings:", e)
        return rows  # Return empty list
        print("Accessing topic now!")
        topic = data.get("topic", "")
        error_categories = data.get("error_categories", [])    

        for category in error_categories:
            label = category.get("label", "")
            pct = category.get("percentage", None)
            examples = category.get("examples", [])
            for example in examples:
                rows.append({
                    "topic": topic,
                    "error_label": label,
                    "percentage": pct,
                    "excerpt": example.get("excerpt", ""),
                    "article_provided": example.get("article_provided", None),
                    "article_failure_reason": example.get("article_failure_reason", ""),
                })

    # Optional: include new categories definitions (as rows without examples)
    # new_categories = data.get("new_categories", [])
    # for new_cat in new_categories:
    #     rows.append({
    #         "topic": topic,
    #         "error_label": new_cat.get("label", ""),
    #         "percentage": None,
    #         "excerpt": "",
    #         "article_provided": None,
    #         "article_failure_reason": "",
    #     })

    return rows

def export_all_taxonomies_to_csv(
    json_file,
    output_csv_path="../data/processed/error_analysis.csv"
):
    """
    Takes a list of JSON strings and appends rows to a CSV.
    """
    all_rows = []
    with open(json_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                rows = parse_taxonomy_json(line)
                all_rows.extend(rows)

    if not all_rows:
        print("⚠️ No rows to export. Skipping.")
        return

    df = pd.DataFrame(all_rows)

    file_exists = os.path.exists(output_csv_path)

    df.to_csv(
        output_csv_path,
        mode="a",  # append mode
        index=False,
        header=not file_exists  # write header only if file doesn't exist
    )
    print(f"✅ Exported {len(df)} rows to {output_csv_path} (append={file_exists})")
    
# def export_all_taxonomies_to_csv(json_strings, output_csv_path="..\data\processed\error_analysis.csv"):
#     """
#     Takes a list of JSON strings and writes a single CSV with all rows.
#     """
#     all_rows = []
#     for json_str in json_strings:
#         rows = parse_taxonomy_json(json_str)
#         all_rows.extend(rows)

#     df = pd.DataFrame(all_rows)
#     df.to_csv(output_csv_path, index=False)
#     print(f"✅ Exported {len(df)} rows to {output_csv_path}")

SUBTOPIC_TO_PARENT = {
    # Leave Management / FMLA
    "-1_leave_fmla_case_status": "Leave Management / FMLA",
    "1_ir_receipt_status_fmla": "Leave Management / FMLA",
    "5_bonding_pfl_bond_ka06e0000011i5gcae": "Leave Management / FMLA",
    "7_certification_healthcare_provider_doctor": "Leave Management / FMLA",
    "15_extension_extend_proxy_continuous": "Leave Management / FMLA",
    "17_mother_care_take_family": "Leave Management / FMLA",
    "18_upload_document_doctor_fmla": "Leave Management / FMLA",
    "33_fmla_eligibility_explore_matrix": "Leave Management / FMLA",
    "46_cfra_act_right_family": "Leave Management / FMLA",
    "75_maternity_cfra_pregnancy_bonding": "Leave Management / FMLA",
    "77_colorado_co_famli_cdle_famli_info": "Leave Management / FMLA",
    "87_center_service_confusion_fmla": "Leave Management / FMLA",
    "92_ka03j000000emclcas_election_vacation_loa": "Leave Management / FMLA",
    "93_washington_dc_pfml_washingtonpaidfamilymedicalleave_default": "Leave Management / FMLA",
    "95_track_log_expiration_fmla": "Leave Management / FMLA",
    "123_pregnancy_maternity_varie_lifeevent": "Leave Management / FMLA",
    "148_doctor_fmla_paperwork_status": "Leave Management / FMLA",
    "152_transition_baby_bonding_maternity": "Leave Management / FMLA",
    "153_fmla_maryland_certification_regions": "Leave Management / FMLA",
    "159_colorado_coloradofamilymedicalleaveinsuranceprogram_coloradofamilymedicalleaveinsurance_default_co": "Leave Management / FMLA",
    "165_maternity_extension_processing_doctor": "Leave Management / FMLA",
    "169_maternity_department_promise_pregnancy": "Leave Management / FMLA",
    "204_extension_doctor_extend_25th": "Leave Management / FMLA",
    "219_maternity_revise_form_seek": "Leave Management / FMLA",

    # Disability & State Claims
    "10_sdi_pfl_ca_family": "Disability & State Claims",
    "31_sdi_edd_ca_waiting": "Disability & State Claims",
    "43_edd_claim_ca_sdi": "Disability & State Claims",
    "96_edd_disability_hotline_website": "Disability & State Claims",
    "119_pfl_edd_www_gov": "Disability & State Claims",
    "138_claim_edd_www_gov": "Disability & State Claims",
    "205_questionnaire_extension_edd_continuous": "Disability & State Claims",
    "207_questionnaire_edd_fraud_claim": "Disability & State Claims",

    # Enrollment & Benefits
    "4_enrollment_dependent_enroll_enrol": "Enrollment & Benefits",
    "13_cobra_equity_coverage_cost": "Enrollment & Benefits",
    "50_plan_coverage_waive_spd": "Enrollment & Benefits",
    "79_mercer_ka03j000000em3pcac_design_deduction": "Enrollment & Benefits",
    "80_cobra_package_healthequity_coverage": "Enrollment & Benefits",
    "83_mercer_deduction_administrator_aflac": "Enrollment & Benefits",
    "100_delta_dental_deltacare_usa": "Enrollment & Benefits",
    "133_parent_medicare_mother_enrol": "Enrollment & Benefits",
    "179_discount_travel_kaiser_plan": "Enrollment & Benefits",
    "193_medicare_part_ka03j000000elyfcac_enroll": "Enrollment & Benefits",
    "198_spouse_mselve_coverage_enrollment": "Enrollment & Benefits",
    "221_ppo_hmo_plan_dentist": "Enrollment & Benefits",

    # Payroll / Compensation
    "2_discrepancy_paycheck_discrepancie_promise": "Payroll / Compensation",
    "6_std_continuance_salary_metlife": "Payroll / Compensation",
    "41_deposit_recall_account_payment": "Payroll / Compensation",
    "45_specialist_increase_raise_coordinate": "Payroll / Compensation",
    "108_cycle_payroll_payment_friday": "Payroll / Compensation",
    "156_copy_payslip_stub_advice": "Payroll / Compensation",
    "157_overpayment_coworker_center_occur": "Payroll / Compensation",
    "175_paycheck_pto_payout_remain": "Payroll / Compensation",
    "178_advance_specialist_ed_salary": "Payroll / Compensation",
    "181_cash_payout_lbd_bsl": "Payroll / Compensation",
    "220_overpayment_repayment_credit_notice": "Payroll / Compensation",

    # Taxes & Withholding
    "58_impute_deduction_income_tax": "Taxes & Withholding",
    "65_tax_withholding_taxis_income": "Taxes & Withholding",
    "180_irs_tax_withholding_lift": "Taxes & Withholding",
    "55_surcharge_area_washington_reside": "Taxes & Withholding",

    # Retirement
    "12_retirement_center_kaiser_pension": "Retirement",
    "30_vanguard_contribution_fidelity_pension": "Retirement",
    "66_retirement_award_pension_saving": "Retirement",
    "69_retirement_kprc_fidelity_pension": "Retirement",

    # Timekeeping & Scheduling
    "35_timecard_timekeepe_clock_recharge": "Timekeeping & Scheduling",
    "89_ka03j000000emhycac_return_block_schedule": "Timekeeping & Scheduling",
    "90_timecard_hdl_load_automate": "Timekeeping & Scheduling",
    "104_query_count_workforce_hour": "Timekeeping & Scheduling",
    "127_timekeepe_correction_coordinate_miss": "Timekeeping & Scheduling",

    # Verification & Documentation
    "11_fax_email_attachment_format": "Verification & Documentation",
    "19_esl_ka03j000000elxacas_hospitalization_outpatient": "Verification & Documentation",
    "23_twn_employment_verification_resend": "Verification & Documentation",
    "40_ssn_birthdate_datum_name": "Verification & Documentation",
    "64_verification_employment_letter_assistance": "Verification & Documentation",
    "129_upload_marriage_certificate_assistance": "Verification & Documentation",
    "143_fax_receipt_document_upload": "Verification & Documentation",
    "161_poa_deadline_sla_validation": "Verification & Documentation",
    "191_baby_birth_certificate_child": "Verification & Documentation",

    # Job Changes & Terminations
    "3_department_transfer_identity_terminate": "Job Changes & Terminations",
    "37_transfer_entity_unit_position": "Job Changes & Terminations",
    "86_promotion_change_data_hrar": "Job Changes & Terminations",
    "105_termination_rescind_resignation_notice": "Job Changes & Terminations",
    "111_termination_letter_june_1st": "Job Changes & Terminations",
    "120_resignation_termination_heuhzce3_v2": "Job Changes & Terminations",
    "162_position_requisition_category_planandhire": "Job Changes & Terminations",

    # Access & Technical Issues
    "160_lcr_license_registration_calendar": "Access & Technical Issues",
    "208_pingid_browser_edge_cache": "Access & Technical Issues",
    "118_ppe_integration_code_base": "Access & Technical Issues",

    # HR General / Operations
    "8_center_service_hr_issue": "HR General / Operations",
    "22_center_service_department_hr": "HR General / Operations",
    "164_directory_consultant_performance_area": "HR General / Operations",
}

def label_sub_topics(csv_store_file, new_file="my_findings.csv"):
    with open(csv_store_file, "r", newline="", encoding="utf-8") as infile, \
         open(new_file, "w", newline="", encoding="utf-8") as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["Parent Topic"]

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            label = row["Label"]
            parent_topic = SUBTOPIC_TO_PARENT.get(label, "Other / Miscellaneous")
            row["Parent Label"] = parent_topic
            writer.writerow(row)