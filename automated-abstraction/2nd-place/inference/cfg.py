class PRED_CFG:
    num_workers=4
    test_path="data/test_features.csv"
    pretrain_model = "assets/flan-t5-xl/"
    path=[
        "assets/deberta_large_manualdesc_1664_f0/",
        "assets/deberta_large_manualdesc_1664_f1/",
        "assets/deberta_large_manualdesc_1664_f2/",
        "assets/deberta_large_manualdesc_1664_f3/",
        "assets/deberta_large_manualdesc_1664_f4/",
        # "assets/deberta_large_2_manualdesc_1664_f0/",
        # "assets/deberta_large_2_manualdesc_1664_f1/",
        # "assets/deberta_large_2_manualdesc_1664_f2/",
        # "assets/deberta_large_2_manualdesc_1664_f3/",
        # "assets/deberta_large_2_manualdesc_1664_f4/",
        # "assets/deberta_large_210394_rm_manualdesc_1664_f0/",
        # "assets/deberta_large_210394_rm_manualdesc_1664_f1/",
        # "assets/deberta_large_210394_rm_manualdesc_1664_f2/",
        # "assets/deberta_large_210394_rm_manualdesc_1664_f3/",
        # "assets/deberta_large_210394_rm_manualdesc_1664_f4/",
        # "assets/deberta_large_210394_manualdesc_1792_f0/",
        # "assets/deberta_large_210394_manualdesc_1792_f1/",
        # "assets/deberta_large_210394_manualdesc_1792_f2/",
        # "assets/deberta_large_210394_manualdesc_1792_f3/",
        # "assets/deberta_large_210394_manualdesc_1792_f4/",
        # f"assets/deberta_large_shuffle_1664_f0/",
        # f"assets/deberta_large_shuffle_1664_f1/",
        # f"assets/deberta_large_shuffle_1664_f2/",
        # f"assets/deberta_large_shuffle_1664_f3/",
        # f"assets/deberta_large_shuffle_1664_f4/",
        "assets/longformer_large_mlmv2_noshuffle_f0/",
        "assets/longformer_large_mlmv2_noshuffle_f1/",
        "assets/longformer_large_mlmv2_noshuffle_f2/",
        "assets/longformer_large_mlmv2_noshuffle_f3/",
        "assets/longformer_large_mlmv2_noshuffle_f4/",
        # "assets/flan_xl_1532_f0/",
        # "assets/flan_xl_1532_f1/",
        # "assets/flan_xl_1532_f2/",
        # # "assets/flan_xl_1532_f3_8536/",
        # "assets/flan_xl_1532_f3/",
        # "assets/flan_xl_1532_f4/",
    ]
    batch_size=4
    flan_batch_size=2

class DEFINE:
    column_dict = {
        "DepressedMood": "Depressed Mood", 
        "MentalIllnessTreatmentCurrnt": "Current Mental Illness Treatment",
        "HistoryMentalIllnessTreatmnt": "History Mental Illness Treatment", 
        "SuicideAttemptHistory": "Suicide Attempt History",
        "SuicideThoughtHistory": "Suicide Thought History", 
        "SubstanceAbuseProblem": "Substance Abuse Problem",
        "MentalHealthProblem": "Mental Health Problem", 
        "DiagnosisAnxiety": "Diagnosis Anxiety",
        "DiagnosisDepressionDysthymia": "Diagnosis Depression Dysthymia", 
        "DiagnosisBipolar": "Diagnosis Bipolar", 
        "DiagnosisAdhd": "Diagnosis Adhd", 
        "IntimatePartnerProblem": "Intimate Partner Problem", 
        "FamilyRelationship": "Family Relationship", 
        "Argument": "Argument",
        "SchoolProblem": "School Problem", 
        "RecentCriminalLegalProblem": "Recent Criminal Legal Problem", 
        "SuicideNote": "Suicide Note", 
        "SuicideIntentDisclosed": "Suicide Intent Disclosed", 
        "DisclosedToIntimatePartner": "Disclosed To Intimate Partner", 
        "DisclosedToOtherFamilyMember": "Disclosed To Other Family Member", 
        "DisclosedToFriend": "Disclosed To Friend", 
        "InjuryLocationType": "Injury Location Type", 
        "WeaponType1": "Weapon Type"
    }

    column_dict_manual = {
        "DepressedMood": "Depressed Mood / Sad / Despondent / Down / Blue / Low / Unhappy", 
        "MentalIllnessTreatmentCurrnt": "Current Mental Illness Treatment",
        "HistoryMentalIllnessTreatmnt": "History of Mental Illness Treatment", 
        "SuicideAttemptHistory": "History of Suicide Attempt",
        "SuicideThoughtHistory": "History of Suicide Thought", 
        "SubstanceAbuseProblem": "Substance Abuse Problem",
        "MentalHealthProblem": "Mental Health Problem", 
        "DiagnosisAnxiety": "Diagnosis Anxiety Disorder",
        "DiagnosisDepressionDysthymia": "Diagnosis Depression Dysthymia", 
        "DiagnosisBipolar": "Diagnosis Bipolar Disorder", 
        "DiagnosisAdhd": "Diagnosis Adhd / Attention Deficit / Hyperactivity Disorder", 
        "IntimatePartnerProblem": "Problem With Intimate Partner", 
        "FamilyRelationship": "Problem With A Family Member", 
        "Argument": "Argument Or Conflict",
        "SchoolProblem": "Problem at or related to school", 
        "RecentCriminalLegalProblem": "Recent Criminal Legal Problem", 
        "SuicideNote": "Left A Suicide Note", 
        "SuicideIntentDisclosed": "Suicide Intent Disclosed", 
        "DisclosedToIntimatePartner": "Suicide Intent Disclosed To Intimate Partner", 
        "DisclosedToOtherFamilyMember": "Suicide Intent Disclosed To Other Family Member", 
        "DisclosedToFriend": "Suicide Intent Disclosed To Friend", 
        "InjuryLocationType": "Injury Location Type", 
        "WeaponType1": "Weapon Type"
    }

    il_dict = {
        1: "House, apartment", 2: "Motor vehicle (excluding school bus and public transportation)",
        3: "Natural area (e.g., field, river, beaches, woods)", 4: "Park, playground, public use area",
        5: "Street/road, sidewalk, alley", 6: "Other"
    }
    wt_dict = {
        1: "Blunt instrument", 2: "Drowning", 3: "Fall", 4: "Fire or burns", 5: "Firearm",
        6: "Hanging, strangulation, suffocation", 7: "Motor vehicle including buses, motorcycles",
        8: "Other transport vehicle, eg, trains, planes, boats", 9: "Poisoning", 10: "Sharp instrument",
        11: "Other (e.g. taser, electrocution, nail gun)", 12: "Unknown"
    }