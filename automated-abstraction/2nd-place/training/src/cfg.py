class CFG:
    apex=False
    manual=True
    data_path="../proc_data/train_5folds.csv"
    encoder_lr=2e-5
    decoder_lr=2e-5
    fold=0
    epochs=5
    seed=42

    max_length=800 # used to be 1664
    output_dir=f"./saved_models/deberta_v3l_{epochs}ep_manualdesc_{seed}_{encoder_lr}_{max_length}_f{fold}/"
    log_file_name=f"logs/deberta_v3l_{epochs}ep_manualdesc_{seed}_{encoder_lr}_{max_length}_f{fold}"
    model_name="allenai/longformer-large-4096" 
    # model_name="microsoft/deberta-v3-large"
    batch_size=1 # decreased from 4
    scheduler="cosine"
    print_freq=200
    num_workers=4
    shuffle=False 
    max_grad_norm=1000
    num_warmup_steps=0
    batch_scheduler=True
    eps=1e-6
    betas=(0.9, 0.999)
    num_cycles=0.5
    weight_decay=0.01

class PRED_CFG:
    num_workers=4
    models={
            "deberta_large_manualdesc_1664": 
                [
                    "./saved_models/deberta_large_manualdesc_1664_f0/",
                    "./saved_models/deberta_large_manualdesc_1664_f1/",
                    "./saved_models/deberta_large_manualdesc_1664_f2/",
                    "./saved_models/deberta_large_manualdesc_1664_f3/",
                    "./saved_models/deberta_large_manualdesc_1664_f4/",
              ],
            "longformer_large_mlmv2_noshuffle":
                [
                    "./saved_models/longformer_large_mlmv2_noshuffle_f0/",
                    "./saved_models/longformer_large_mlmv2_noshuffle_f1/",
                    "./saved_models/longformer_large_mlmv2_noshuffle_f2/",
                    "./saved_models/longformer_large_mlmv2_noshuffle_f3/",
                    "./saved_models/longformer_large_mlmv2_noshuffle_f4/",
   
                ]
            }
    batch_size=32
    seed=42


class DEFINE:
    column_manual_dict = {
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