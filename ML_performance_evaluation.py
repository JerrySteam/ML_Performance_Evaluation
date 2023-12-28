from module import *

# Project/dataset directory path
main_path = "C:/Users/Jerry/Desktop/Implementation/"
dataset_parasite = "parasite/plasmodium"
dataset_bacteria = "bacteria/bacillus"
project_type = "project/ml_performance_evaluation"

project_type_with_underscore = str(project_type.split("/")[0]) + "_" + str(project_type.split("/")[1])
file_tag = str(time.time()).split(".")[0]

# create temporary directory for storing files if it does not exist
tmp_dir = createDir(main_path+project_type+"/"+"tmp/")

# ===================Data Preprocessing starts here =============================
print("Preprocessing Starts...")
dataset_ls = [dataset_parasite]
hpi_data = pd.DataFrame()
for dataset in dataset_ls:
    # read dataset files (fasta format) from the project directory into list
    int_host = readFasta(main_path+dataset+"/"+"inthost.fasta")
    int_pathogen = readFasta(main_path+dataset+"/"+"intpathogen.fasta")
    nonint_host = readFasta(main_path+dataset+"/"+"noninthost.fasta")
    nonint_pathogen = readFasta(main_path+dataset+"/"+"nonintpathogen.fasta")

    # delete any invalid protein sequence
    print("Removing invalid protein pairs...")
    int_host, int_pathogen = deleteInvalidProt(int_host, int_pathogen, "interacting")
    nonint_host, nonint_pathogen = deleteInvalidProt(nonint_host, nonint_pathogen, "noninteracting")

    # set output file path for the feature vectors
    int_host_path = tmp_dir+"int_host_"+file_tag
    int_pathogen_path = tmp_dir+"int_pathogen_"+file_tag
    nonint_host_path = tmp_dir+"nonint_host_"+file_tag
    nonint_pathogen_path = tmp_dir+"nonint_pathogen_"+file_tag

    # perform the feature encoding technique (Note: This operations takes time)
    print("Performing protein sequence feature encoding...")
    int_hostx_path = generateFeatures(
        convertToFasta(int_host, int_host_path+".fasta"), int_host_path+"_x.csv")
    int_pathogenx_path = generateFeatures(
        convertToFasta(int_pathogen, int_pathogen_path+".fasta"), int_pathogen_path+"_x.csv")
    nonint_hostx_path = generateFeatures(
        convertToFasta(nonint_host, nonint_host_path+".fasta"), nonint_host_path+"_x.csv")
    nonint_pathogenx_path = generateFeatures(
        convertToFasta(nonint_pathogen, nonint_pathogen_path+".fasta"), nonint_pathogen_path+"_x.csv")

    # Remove index column from files
    int_hostx = pd.read_csv(int_hostx_path, delimiter='\t', encoding='latin-1').iloc[: , 1:]
    int_pathogenx = pd.read_csv(int_pathogenx_path, delimiter='\t', encoding='latin-1').iloc[: , 1:]
    nonint_hostx = pd.read_csv(nonint_hostx_path, delimiter='\t', encoding='latin-1').iloc[: , 1:]
    nonint_pathogenx = pd.read_csv(nonint_pathogenx_path, delimiter='\t', encoding='latin-1').iloc[: , 1:]

    # combine host and pathogen protein features and add label feature
    int_hp = pd.concat([int_hostx, int_pathogenx], axis=1)
    nonint_hp = pd.concat([nonint_hostx, nonint_pathogenx], axis=1)
    int_hp['label'] = 1
    nonint_hp['label'] = 0

    enc_type = "aap"

    # combine interacting and non interacting protein
    hpi_int_nonint = pd.concat([int_hp, nonint_hp], axis=0, ignore_index=True)
    print(hpi_int_nonint.shape)
    print(hpi_int_nonint['label'].value_counts())

    # merge previous data with the existing data i.e all organism datasets
    hpi_data = pd.concat([hpi_data, hpi_int_nonint], axis=0, ignore_index=True)

# save extracted features to csv
preprocessedPath = createDir(main_path+project_type+"/"+ "preprocessed/")
hpi_data.to_csv(preprocessedPath + project_type_with_underscore + ".csv")
print("Preprocessing Completed!")

print(hpi_data.shape)
print(hpi_data['label'].value_counts())
# ===================end of data preprocessing ==============================

# ===================Computational model starts =============================
# hpi_data = pd.read_csv(main_path+project_type+"/"+"preprocessed/" + project_type_with_underscore + ".csv", index_col=[0])

print("\n Model Training Starts...")
# count number of interactions and non-interactions
print(hpi_data['label'].value_counts())
print(hpi_data.shape)

# Output path directory
outputPath = createDir(main_path+project_type+"/"+ "output/")

# Create output file in excel
workbook = xlsxwriter.Workbook(outputPath + project_type_with_underscore + file_tag + ".xlsx")
worksheet = workbook.add_worksheet()

# Run experiment for both balanced and imbalanced dataset i.e range(0,2).
# If only balance, set range(0,1), if only imbalance, set range (1,2)
j = 0
for count in range(0, 1):
    if count == 0: analysis_type = "balanced"
    else: analysis_type = "imbalanced"
    fileTitle = str(analysis_type) + "_dataset_" + str(project_type_with_underscore)

    if analysis_type == "balanced":
        # separate interacting proteins from non-interacting ones
        int_hpi_data = hpi_data[hpi_data['label'] == 1]
        nonint_hpi_data = hpi_data[hpi_data['label'] == 0]

        # print("\nInteracting protein label count")
        print("\nBalanced dataset label count")
        if len(int_hpi_data) >= len(nonint_hpi_data):
            # Create balance dataset: Use this when interacting is more than non-interacting samples
            # randomly select interacting samples
            rand_int_hpi_data = int_hpi_data.sample(len(nonint_hpi_data))

            # merge the nonint random samples with the int samples to form a balance dataset
            bal_hpi_data = pd.concat([rand_int_hpi_data, nonint_hpi_data], axis=0, ignore_index=True)

            # separate balanced hpi features from labels
            X_hpi = bal_hpi_data.drop(columns='label')
            y_hpi = bal_hpi_data.label
            print(y_hpi.value_counts())

        else:
            # Create balance dataset: Use this when non-interacting is more than interacting samples
            # randomly select non-interacting samples
            rand_nonint_hpi_data = nonint_hpi_data.sample(len(int_hpi_data), replace=False)

            # merge the nonint random samples with the int samples to form a balance dataset
            bal_hpi_data = pd.concat([int_hpi_data, rand_nonint_hpi_data], axis=0, ignore_index=True)

            # separate balanced hpi features from labels
            X_hpi = bal_hpi_data.drop(columns='label')
            y_hpi = bal_hpi_data.label
            print(y_hpi.value_counts())

    else:
        # Create imbalance dataset
        # separate imbalanced hpi features from labels
        X_hpi = hpi_data.drop(columns='label')
        y_hpi = hpi_data.label
        # print("\nNon-interacting protein label count")
        print("\nImbalanced dataset label count")
        print(y_hpi.value_counts())

    # The classifiers that would be trained
    clfs = [
        {'label': 'RF', 'model': RandomForestClassifier()},
        {'label': 'SVM', 'model': SVC()},
        {'label': 'MLP', 'model': MLPClassifier()},
        {'label': 'NB', 'model': GaussianNB()},
        {'label': 'LR', 'model': LogisticRegression()},
        {'label':'DF', 'model': CascadeForestClassifier()},
    ]

    # Output file header in excel
    worksheet.write(j, 0, fileTitle)
    worksheet.write(j + 1, 1, "Accuracy")
    worksheet.write(j + 1, 2, "Sensitivity")
    worksheet.write(j + 1, 3, "Specificity")
    worksheet.write(j + 1, 4, "Precision")
    worksheet.write(j + 1, 5, "F1 Score")
    worksheet.write(j + 1, 6, "MCC")
    worksheet.write(j + 1, 7, "AUROC")
    worksheet.write(j + 1, 8, "Time (sec)")
    worksheet.write(j + 1, 9, "Space (mb)")

    np.random.seed(seed)
    all_model_performance = []
    for clf in clfs:
        model_performance = []
        print(clf['label'] + " Model: RUNNING....")
        # Capture time and memory stating point
        tracemalloc.start()
        start_time = time.time()

        # replace NAN. INFINITY to Zero
        X_hpi = np.nan_to_num(X_hpi)

        # Define the model and scoring metrics
        model = make_pipeline(MinMaxScaler(), clf['model'])
        scoring = {
            'accuracy': make_scorer(custom_scorer, custom_metric='accuracy'),
            'sensitivity': make_scorer(custom_scorer, custom_metric='sensitivity'),
            'specificity': make_scorer(custom_scorer, custom_metric='specificity'),
            'precision': make_scorer(custom_scorer, custom_metric='precision'),
            'f1': make_scorer(custom_scorer, custom_metric='f1'),
            'mcc': make_scorer(custom_scorer, custom_metric='mcc'),
            'auroc': make_scorer(custom_scorer, custom_metric='auroc')
        }

        # Perform 5-fold cross-validation with scoring metrics
        cv_results = cross_validate(model, X_hpi, y_hpi, cv=1, scoring=scoring)

        # Capture time and memory end point and measure the difference
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Measure predictive performance. Metrics to be measured are:
        # Accuracy, Sensitivity, Specificity, Precision, F1 Score, MCC and AUROC.
        # In addition, time and memory usage would also be measured

        accuracy = cv_results['test_accuracy'].mean()
        sensitivity = cv_results['test_sensitivity'].mean()
        specificity = cv_results['test_specificity'].mean()
        precision = cv_results['test_precision'].mean()
        f1_score = cv_results['test_f1'].mean()
        mcc = cv_results['test_mcc'].mean()
        auroc = cv_results['test_auroc'].mean()

        worksheet.write(j + 2, 0, clf['label'])
        worksheet.write(j + 2, 1, accuracy)
        worksheet.write(j + 2, 2, sensitivity)
        worksheet.write(j + 2, 3, specificity)
        worksheet.write(j + 2, 4, precision)
        worksheet.write(j + 2, 5, f1_score)
        worksheet.write(j + 2, 6, mcc)
        worksheet.write(j + 2, 7, auroc)
        worksheet.write(j + 2, 8, end_time - start_time)
        worksheet.write(j + 2, 9, current / 10 ** 6)

        model_performance = [accuracy, sensitivity, specificity, precision, f1_score, mcc, auroc]
        all_model_performance.append(model_performance)
        print(f"{clf['label']}: {model_performance}")

        j = j + 1
        print(clf['label'] + " Model: FINISHED RUNNING!\n")
    j = j + 3
workbook.close()