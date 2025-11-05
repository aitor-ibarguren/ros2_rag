# RENEE Product State Diagnosis Module

This repository contains the Product State Diagnosis module of RENEE project. It contains the next packages:

* **renee_product_state_diagnosis_module:** The Product State Diagnosis module node with all its functionalties. The node implements a Lifecycle Node in Python using **Pandas** and **Scikit-Learn** libraries for data management and classification.
* **renee_product_state_diagnosis_module_msgs:** Messages of the Product State Diagnosis module.

The node offers different ROS2 services to load and manage data, as well as classify new cases. Next lines provide further information about the node services, as well as input data format.

## RENEE PDS Module services

The Product State Diagnosis module node offers the next services:

* **/renee_psd_node/load_data:** Service to load data files containing product and core information. This loaded data is further used to train prediction models. The [next section](#info-and-data-file-format) provides further information about the file format accepted by the PSD module.
* **/renee_psd_node/train_model:** This service creates prediction models based on the loaded data. The PSD module generates a separate classifier for each of the cores.
* **/renee_psd_node/predict:** This service generates a query to predict the state of a specific core based on the provided data.
* **/renee_psd_node/predict_all:** This service generates a query to predict all cores' state based on the provided data.

## Info and Data file format

For the generation of prediction models, the PSD requires two different input files:

* A header or information file (e.g. *emotors.info*) with metadata about the product and its' cores in YAML format. Specifically, the file includes the next fileds:
  * **columns** indicating the number of columns of the data file.
  * For each column, it is necessary to indicate its **ID**, if it is a **data** or **core**, and the **format** (*Category* or *Double*). In the case of the *category* format, it is also required to include a **tags** field to define the different options.

```yaml
some:
columns: 10
column_1:
  id: "Connector interface inspection"
  type: Data
  format: Category
  tags: ["correct","damaged"]
column_2:
  id: "Shaft inspection"
  type: Data
  format: Category
  tags: ["correct","rusty","damaged"]
column_3:
  id: "Overall inspection"
  type: Data
  format: Category
  tags: ["correct","damaged","useless"]
column_4:
  id: "Vibratory and torque resistance"
  type: Data
  format: Double
column_5:
  id: "Sealing test"
  type: Data
  format: Category
  tags: ["sealed","fuild loss"]
column_6:
  id: "Torque test"
  type: Data
  format: Double
column_7:
  id: "Speed test"
  type: Data
  format: Double
column_8:
  id: "Rotor"
  type: Core
  format: Category
  tags: ["second life","second life non automotive","recycling"]
column_9:
  id: "Stator"
  type: Core
  format: Category
  tags: ["second life","second life non automotive","recycling"]
column_10:
  id: "Housing"
  type: Core
  format: Category
  tags: ["second life","second life non automotive","recycling"]
```

* A data file is CSV format using commas to separate the fields. During the loading phase, the PSD module will check the consistency of the data based on the information included in the header (INFO) file.
