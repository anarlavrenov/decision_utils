import string

CHECKPOINT = "Qwen/Qwen3.8-27B"
HF_REPO_SAVE_PATH = "anarlavrenov/Decision-Qwen3.8-27B"
MODAL_APP_NAME = "decision-qwen-27b"
CHECKPOINT_DIR = "/checkpoints"
GRAD_ACCUM_STEPS = 4
BATCH_SIZE = 8
N_EPOCHS = 1
N_TRAIN_SAMPLES_PER_DATASET = 50_000
N_TRAIN_SAMPLES_JEV_PER_DATASET = 150_000
N_VALID_SAMPLES_PER_DATASET = 300
MAX_CHARS = 16000
PAD_ID = 248044
LORA_RANK = 16
LORA_ALPHA = 32
LEARNING_RATE = 2e-4
LR_PERCENTAGE_START = 0.1
LOG_EVERY_TRAIN = 1000
LOG_EVERY_VALID = 5000
SEED = 0

# HF Datasets names.
HF_JEV_TYPED_DECISION_PATH = "tasksource/tasksource-jev-typed-decisions"
HF_OPEN_JEV_PATH = "ZefanCai/Open-Jev-v1.1"
HF_LEDGAR_PATH = "coastalcph/lex_glue"
HF_LEDGAR_SUBSET_NAME = "ledgar"
HF_CASEHOLD_SUBSET_NAME = "case_hold"
HF_NLI_PATH = "kiddothe2b/contract-nli"
HF_NLI_REVISION_NAME = "refs/convert/parquet"
HF_ANLI_PATH = "facebook/anli"
HF_ANLI_SPLIT_NAME = "train_r1"
HF_MMLU_PATH = "cais/mmlu"
HF_MMLU_SUBSET_NAME = "auxiliary_train"
HF_SCIQ_PATH = "allenai/sciq"
HF_HELLASWAG_PATH = "Rowan/hellaswag"
HF_VITAMIN_PATH = "tals/vitaminc"

TRAIN_SPLIT_NAME = "train"
VAL_SPLIT_NAME = "validation"

LETTERS = string.ascii_uppercase
LEDGAR_CLASS_DESCRIPTIONS = {
    "Adjustments": "Specifies how contract amounts, prices, or other terms may be recalculated.",
    "Agreements": "Sets out the parties' contractual commitments and agreed obligations.",
    "Amendments": "Defines how an existing agreement may be formally amended.",
    "Anti-Corruption Laws": "Requires compliance with anti-bribery and anti-corruption laws.",
    "Applicable Laws": "Identifies laws and regulations applicable to the parties or transaction.",
    "Approvals": "Specifies approvals required to authorize actions or complete a transaction.",
    "Arbitration": "Defines procedures for resolving disputes through arbitration.",
    "Assignments": "Governs transfers of contractual rights or obligations to third parties.",
    "Assigns": "Specifies how contractual rights and obligations extend to permitted assignees.",
    "Authority": "Confirms a party's legal power to enter into and perform the agreement.",
    "Authorizations": "Addresses legal or organizational authorizations required for contractual actions.",
    "Base Salary": "Sets an employee's fixed compensation and conditions for its payment.",
    "Benefits": "Specifies employee benefit plans, eligibility, and entitlements.",
    "Binding Effects": "States that the agreement is binding on the parties and their successors or assigns.",
    "Books": "Governs the maintenance and inspection of corporate books and accounting records.",
    "Brokers": "Addresses broker or finder involvement and related commissions.",
    "Capitalization": "Describes a company's equity structure and outstanding securities.",
    "Change In Control": "Defines changes in ownership or control and their contractual consequences.",
    "Closings": "Sets procedures and requirements for completing a transaction.",
    "Compliance With Laws": "Requires parties to conduct activities in accordance with applicable laws.",
    "Confidentiality": "Restricts the use and disclosure of confidential information.",
    "Consent To Jurisdiction": "Records a party's consent to specified courts hearing disputes.",
    "Consents": "Specifies third-party or governmental consents required under the agreement.",
    "Construction": "Establishes rules for interpreting contractual language and provisions.",
    "Cooperation": "Requires parties to assist one another in fulfilling contractual obligations.",
    "Costs": "Allocates costs arising from the agreement or transaction.",
    "Counterparts": "Permits execution in separate copies that together constitute one agreement.",
    "Death": "Defines the consequences of an individual's death for contractual rights or benefits.",
    "Defined Terms": "Identifies contractual terms assigned specific meanings throughout the agreement.",
    "Definitions": "Provides definitions of words and expressions used in the agreement.",
    "Disability": "Defines disability and its effects on employment, benefits, or contractual rights.",
    "Disclosures": "Specifies information that must be disclosed or provided to other parties.",
    "Duties": "Describes responsibilities and obligations assigned to a party or employee.",
    "Effective Dates": "Specifies when the agreement or a contractual provision takes effect.",
    "Effectiveness": "Establishes conditions under which the agreement becomes legally effective.",
    "Employment": "Governs the employment relationship and its essential conditions.",
    "Enforceability": "Confirms that contractual obligations are valid and legally enforceable.",
    "Enforcements": "Describes procedures for enforcing contractual rights and obligations.",
    "Entire Agreements": "States that the written agreement supersedes prior agreements on the same subject.",
    "Erisa": "Addresses compliance with ERISA and requirements concerning employee benefit plans.",
    "Existence": "Confirms an entity's legal existence and continued organizational status.",
    "Expenses": "Specifies which expenses are reimbursable or borne by each party.",
    "Fees": "Establishes service, transaction, or other fees and their payment conditions.",
    "Financial Statements": "Governs the preparation, accuracy, and delivery of financial statements.",
    "Forfeitures": "Defines circumstances in which contractual rights, benefits, or awards are lost.",
    "Further Assurances": "Requires additional documents or actions necessary to fulfill the agreement.",
    "General": "Contains general provisions governing the parties' contractual relationship.",
    "Governing Laws": "Identifies the law governing interpretation and enforcement of the agreement.",
    "Headings": "Clarifies whether section headings affect interpretation of contractual provisions.",
    "Indemnifications": "Requires compensation for specified losses, claims, damages, or liabilities.",
    "Indemnity": "Defines obligations to protect another party against specified losses or liabilities.",
    "Insurances": "Specifies required insurance coverage, policies, and related obligations.",
    "Integration": "Establishes the agreement as the complete and final expression of the parties' understanding.",
    "Intellectual Property": "Allocates ownership, licensing, and permitted use of intellectual property.",
    "Interests": "Governs interest accrual, rates, payments, or financial interests under the agreement.",
    "Interpretations": "Specifies principles for interpreting contractual terms and resolving ambiguities.",
    "Jurisdictions": "Identifies courts or legal jurisdictions authorized to hear contractual disputes.",
    "Liens": "Governs security interests, encumbrances, or claims against assets.",
    "Litigations": "Addresses existing or potential lawsuits and related legal proceedings.",
    "Miscellaneous": "Contains supplementary contractual provisions not covered by specific sections.",
    "Modifications": "Regulates changes to existing contractual terms, conditions, or obligations.",
    "No Conflicts": "Confirms that entering into the agreement does not violate other obligations or laws.",
    "No Defaults": "Confirms the absence of defaults or breaches under relevant agreements.",
    "No Waivers": "States that failure or delay in exercising rights does not constitute their waiver.",
    "Non-Disparagement": "Restricts parties from making statements that damage another party's reputation.",
    "Notices": "Specifies how formal communications and notifications must be delivered.",
    "Organizations": "Describes an entity's legal formation, structure, or organizational status.",
    "Participations": "Governs participation interests in loans, investments, or contractual rights.",
    "Payments": "Defines payment amounts, schedules, methods, and related obligations.",
    "Positions": "Specifies an employee's role, position, or responsibilities within an organization.",
    "Powers": "Grants or limits authority to perform actions or exercise specified powers.",
    "Publicity": "Regulates public announcements, communications, and disclosure of contractual relationships.",
    "Qualifications": "Confirms an entity's legal qualification or eligibility to conduct business.",
    "Records": "Governs the creation, maintenance, retention, or inspection of records.",
    "Releases": "Discharges parties from specified claims, obligations, or liabilities.",
    "Remedies": "Defines legal or contractual relief available following a breach or default.",
    "Representations": "Sets out factual statements made by parties concerning their circumstances or obligations.",
    "Sales": "Governs the sale or transfer of assets, goods, or securities.",
    "Sanctions": "Requires compliance with economic, financial, or trade sanctions restrictions.",
    "Severability": "Preserves remaining contractual provisions if a particular provision is invalid.",
    "Solvency": "Addresses a party's financial ability to meet its obligations and liabilities.",
    "Specific Performance": "Allows a party to seek a court order compelling performance of contractual duties.",
    "Submission To Jurisdiction": "Requires parties to submit to the authority of designated courts.",
    "Subsidiaries": "Addresses subsidiary entities and their relationships, rights, or obligations.",
    "Successors": "Specifies how contractual rights and obligations extend to legal successors.",
    "Survival": "Identifies contractual obligations that remain effective after termination or expiration.",
    "Tax Withholdings": "Specifies taxes that must be deducted or withheld from contractual payments.",
    "Taxes": "Allocates responsibility for taxes and establishes related tax obligations.",
    "Terminations": "Defines grounds, procedures, and consequences for ending an agreement or employment.",
    "Terms": "Establishes the duration or essential conditions of a contractual arrangement.",
    "Titles": "Addresses legal ownership or title to assets, property, or securities.",
    "Transactions With Affiliates": "Regulates transactions between parties and their affiliated entities.",
    "Use Of Proceeds": "Restricts or specifies how funds received under the agreement may be used.",
    "Vacations": "Defines employee vacation entitlements, accrual, and permitted leave.",
    "Venues": "Specifies the location or court where legal proceedings must take place.",
    "Vesting": "Establishes when ownership rights in equity, compensation, or benefits become vested.",
    "Waiver Of Jury Trials": "Records an agreement to resolve legal disputes without a jury trial.",
    "Waivers": "Specifies contractual rights, demands, or procedural protections voluntarily relinquished.",
    "Warranties": "Provides contractual assurances concerning the condition, quality, or accuracy of specified matters.",
    "Withholdings": "Governs amounts deducted or withheld from payments under the agreement.",
}
