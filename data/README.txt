                       OHNLP Challenge 2018 Task 2
                    Clinical Semantic Textual Similarity (ClinicalSTS) 

This package contains the training data for the Clinical Semantic Textual Similarity
(ClinicalSTS) shared task. The dataset has the following tab-separated format:

  * One STS pair per line.
  * Each line contains the following fields: STS Sent1[tab]STS Sent2[tab]Similarity Score


Example:

Insulin NPH Human [NOVOLIN N] 100 unit/mL suspension subcutaneous as directed by prescriber.	 Insulin NPH Human [NOVOLIN N] 100 unit/mL suspension 63-76 units subcutaneous as directed by prescriber.	3.5 

-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
Note: Similarity score is a continuous value on a scale from 0 to 5, with 0 indicating that the medical semantics
 of the sentences are completely independent and 5 signifying semantic equivalence.  
correlation-noconfidence.pl is an evaluate tool for computing Pearson correlation coefficient.
-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

Submission Format: 
-------------
For the testing data, please generate a plain text output file of the form
ClinicalSTS2018.OUTPUT.$TEAM_NAME.$RunName.txt

Example: ClinicalSTS2018.OUTPUT.MyTeam.LSTM-Run.txt

The evaluation file should have one line for each STS pair that provides the 
score assigned by your system as a floating point number:

0.1
4.9
3.5
2.0

By default, each team can submit up to three runs. Teams that feel they have a 
good reason for submitting more than three runs should contact the organizers.
Such requests will be handled on a case by case basis.

Good Luck!
