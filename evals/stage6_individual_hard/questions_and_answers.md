# Stage6 Individual Hard - Questions & Answers (MMLU-Pro Hard)

Total questions: 200

Source: TIGER-Lab/MMLU-Pro (HuggingFace) — up to 10-option multiple choice

---

## Q1. In Chord, assume the size of the identifier space is 16. The active nodes are N3, N6, N8 and N12. Show all the target key (in ascending order, ignore the node's identifier itself) for N6.

- **A.** [7, 10, 13, 14]
- **B.** [7, 9, 12, 15]
- **C.** [7, 8, 10, 14]
- **D.** [7, 8, 12, 14]
- **E.** [7, 9, 11, 13]
- **F.** [6, 8, 10, 14]
- **G.** [6, 7, 9, 11]
- **H.** [8, 9, 11, 13]
- **I.** [3, 6, 8, 12]
- **J.** [8, 10, 12, 14]

**Answer: C**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_001*

---

## Q2. Statement 1| Since the VC dimension for an SVM with a Radial Base Kernel is infinite, such an SVM must be worse than an SVM with polynomial kernel which has a finite VC dimension. Statement 2| A two layer neural network with linear activation functions is essentially a weighted combination of linear separators, trained on a given dataset; the boosting algorithm built on linear separators also finds a combination of linear separators, therefore these two algorithms will give the same result.

- **A.** False, False
- **B.** False, True
- **C.** Neither True nor False, False
- **D.** Neither True nor False, True
- **E.** True, True
- **F.** False, Neither True nor False
- **G.** Neither True nor False, Neither True nor False
- **H.** True, False
- **I.** True, Neither True nor False
- **J.** True, False and False, True

**Answer: A**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_002*

---

## Q3. Two single-user workstations are attached to the same local area network. On one of these workstations, file pages are accessed over the network from a file server; the average access time per page is 0.1 second. On the other of these workstations, file pages are accessed from a local disk; the average access time per page is 0.05 second. A particular compilation requires 30 seconds of computation and 200 file page accesses. What is the ratio of the total time required by this compilation if run on the diskless (file server) workstation to the total time required if run on the workstation with the local disk, if it is assumed that computation is not overlapped with file access?

- **A.** 5/4
- **B.** 10/5
- **C.** 6/5
- **D.** 5/3
- **E.** 3/1
- **F.** 4/3
- **G.** 1/1
- **H.** 7/4
- **I.** 3/2
- **J.** 2/1

**Answer: A**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_003*

---

## Q4. Determine the output of the following program READ (2, 3) I, J 3FORMAT (2I11) IF (I - J) 5,6,6 6WRITE (3,10) I 10FORMAT (1X, I11) GO TO 7 5WRITE (3,10) J 7STOP

- **A.** Prints the absolute value of the difference of two numbers
- **B.** Prints out the result of multiplying the two numbers
- **C.** Prints the second number twice
- **D.** Calculates the difference of two numbers
- **E.** Calculates the sum of two numbers
- **F.** Prints out the largest of two numbers
- **G.** Exits without printing anything
- **H.** Prints out the smallest of two numbers
- **I.** Prints out the result of dividing the first number by the second
- **J.** Prints both numbers on the same line

**Answer: F**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_004*

---

## Q5. In the procedure Mystery below, the parameter number is a positive integer.

 PROCEDURE Mystery (number)
 {
  REPEAT UNTIL (number <= 0)
  {
   number ← number - 2
  }
  IF (number = 0)
  {
   RETURN (true)
  }
  ELSE
  {
   RETURN (false)
  }
 }

 Which of the following best describes the result of running the procedure Mystery?

- **A.** The procedure returns true when the initial value of number is a multiple of 2 or 3, and it otherwise returns false.
- **B.** The procedure returns false when the initial value of number is greater than 2, and it otherwise returns true.
- **C.** The procedure returns false when the initial value of number is a prime number, and it otherwise returns true.
- **D.** The procedure returns true when the initial value of number is odd, and it otherwise returns false.
- **E.** The procedure returns false when the initial value of number is even, and it otherwise returns true.
- **F.** The procedure returns true when the initial value of number is less than 2, and it otherwise returns false.
- **G.** The procedure returns true when the initial value of number is even, and it otherwise returns false.
- **H.** The procedure returns true when the initial value of number is 2, and it otherwise returns false.
- **I.** The procedure returns true when the initial value of number is a prime number, and it otherwise returns false.
- **J.** The procedure returns true when the initial value of number is greater than 2, and it otherwise returns false.

**Answer: G**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_005*

---

## Q6. Let a undirected graph G with edges E = {<0,1>,<1,3>,<0,3>,<3,4>,<0,4>,<1,2>,<2,5>,<2,7>,<2,6>,<6,7>,<6,10>,<5,8>,<10,9>,<5,10>,<6,8>,<7,8>,<6,9>,<7,10>,<8,10>,<9,11>,<9,12>,<9,13>,<13,12>,<13,11>,<11,14>}, which <A,B> represent Node A is connected to Node B. What is the shortest path from node 1 to node 14? Represent the path as a list.

- **A.** [1, 2, 7, 8, 10, 9, 11, 14]
- **B.** [1, 2, 5, 10, 6, 9, 11, 14]
- **C.** [1, 3, 0, 4, 2, 6, 9, 11, 14]
- **D.** [1, 2, 5, 8, 10, 14]
- **E.** [1, 2, 7, 6, 8, 10, 9, 11, 14]
- **F.** [1, 3, 4, 0, 2, 5, 8, 10, 9, 11, 14]
- **G.** [1, 0, 3, 2, 5, 8, 9, 11, 14]
- **H.** [1, 3, 4, 0, 2, 6, 9, 11, 14]
- **I.** [1, 0, 4, 3, 2, 7, 10, 9, 11, 14]
- **J.** [1, 2, 6, 9, 11, 14]

**Answer: J**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_006*

---

## Q7. In multiprogrammed systems it is advantageous if some programs such as editors and compilers can be shared by several users. Which of the following must be true of multiprogrammed systems in order that a single copy of a program can be shared by several users?
I. The program is a macro.
II. The program is recursive.
III. The program is reentrant.

- **A.** I and III only
- **B.** II and III are true but not necessary for sharing a program
- **C.** None of the statements must be true
- **D.** II and III only
- **E.** I only
- **F.** I, II, and III all must be true
- **G.** I and II only
- **H.** II only
- **I.** The program is non-recursive and reentrant
- **J.** III only

**Answer: J**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_007*

---

## Q8. What is the output of the following program? PROGRAM TEST (output); VAR I, J, K, COUNT : integer; BEGIN I: = 1; J: = 2; K: = 3; count: = 1; write1n (I div J, K mod J); while count < = 3 DO BEGIN If I < J then If K > J then If (K > I) or (I = J) then write1n (I) ELSE write1n (J) ELSE write1n (K); CASE COUNT OF 1: J: = 0; 2: BEGIN K: = K - 1; J: = J + K END; 3:I: = 2 \textasteriskcentered I; END; count : = count + 1; END {while count........} END.

- **A.** 2 2
- **B.** 2 3
- **C.** 1 0
- **D.** 0 2
- **E.** 3 2
- **F.** 0 1
- **G.** 2 1
- **H.** 3 0
- **I.** 1 1
- **J.** 1 2

**Answer: J**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_008*

---

## Q9. What is the number of labelled forests on 8 vertices with 5 connected components, such that vertices 1, 2, 3, 4, 5 all belong to different connected components?

- **A.** 620
- **B.** 520
- **C.** 220
- **D.** 120
- **E.** 820
- **F.** 320
- **G.** 420
- **H.** 920
- **I.** 720
- **J.** 280

**Answer: F**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_009*

---

## Q10. To compute the matrix product M_1M_2, where M_1 has p rows and q columns and where M_2 has q rows and r columns, takes time proportional to pqr, and the result is a matrix of p rows and r columns. Consider the product of three matrices N_1N_2N_3 that have, respectively, w rows and x columns, x rows and y columns, and y rows and z columns. Under what condition will it take less time to compute the product as (N_1N_2)N_3 (i.e., multiply the first two matrices first) than to compute it as N_1(N_2 N_3)?

- **A.** w > z
- **B.** 1/w + 1/y > 1/x + 1/z
- **C.** y > z
- **D.** 1/x + 1/z < 1/w + 1/y
- **E.** 1/w + 1/z > 1/x + 1/y
- **F.** There is no such condition; i.e., they will always take the same time.
- **G.** x > y
- **H.** 1/w + 1/x < 1/y + 1/z
- **I.** 1/x + 1/y > 1/w + 1/z
- **J.** x < y

**Answer: D**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_010*

---

## Q11. In building a linear regression model for a particular data set, you observe the coefficient of one of the features having a relatively high negative value. This suggests that

- **A.** The model will perform better without this feature
- **B.** This feature does not have a strong effect on the model (should be ignored)
- **C.** It is not possible to comment on the importance of this feature without additional information
- **D.** This feature has a strong effect on the model (should be retained)
- **E.** This feature negatively impacts the model (should be removed)
- **F.** The model is overfitting due to this feature
- **G.** This feature has a weak negative correlation with the target variable
- **H.** The negative coefficient value is a sign of data leakage
- **I.** Nothing can be determined.
- **J.** This feature has a strong positive correlation with the target variable

**Answer: C**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_011*

---

## Q12. At time 0, five jobs are available for execution on a single processor, with service times of 25, 15, 5, 3, and 2 time units. Which of the following is the minimum value of the average completion time of these jobs?

- **A.** 40
- **B.** 10
- **C.** 76/5
- **D.** 100/5
- **E.** 60
- **F.** 208/5
- **G.** 92/5
- **H.** 30
- **I.** 20
- **J.** 50

**Answer: G**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_012*

---

## Q13. We are training fully connected network with two hidden layers to predict housing prices. Inputs are $100$-dimensional, and have several features such as the number of square feet, the median family income, etc. The first hidden layer has $1000$ activations. The second hidden layer has $10$ activations. The output is a scalar representing the house price. Assuming a vanilla network with affine transformations and with no batch normalization and no learnable parameters in the activation function, how many parameters does this network have?

- **A.** 120010
- **B.** 111110
- **C.** 100121
- **D.** 112110
- **E.** 110010
- **F.** 110020
- **G.** 122121
- **H.** 110011
- **I.** 111021
- **J.** 130021

**Answer: I**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_013*

---

## Q14. Statement 1| Traditional machine learning results assume that the train and test sets are independent and identically distributed. Statement 2| In 2017, COCO models were usually pretrained on ImageNet.

- **A.** True, False
- **B.** False, Not stated
- **C.** False, False
- **D.** False, True
- **E.** Not stated, False
- **F.** True, Not stated
- **G.** True, Not applicable
- **H.** True, True
- **I.** Not stated, True
- **J.** Not stated, Not stated

**Answer: H**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_014*

---

## Q15. In a CSMA/CD network with a data rate of 10 Mbps, the minimum frame size is found to be 512 bits for the correct operation of the collision detection process. What should be the minimum frame size (in bits) if we increase the data rate to 1 Gbps?

- **A.** 25600
- **B.** 10000
- **C.** 102400
- **D.** 32000
- **E.** 51200
- **F.** 6400
- **G.** 204800
- **H.** 256000
- **I.** 409600
- **J.** 12800

**Answer: E**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_015*

---

## Q16. An Insect class is to be written, containing the following data fields: age, which will be initialized to 0 when an Insect is constructed. nextAvailableID, which will be initialized to 0 outside the constructor and incremented each time an Insect is constructed. idNum, which will be initialized to the current value of nextAvailableID when an Insect is constructed. position, which will be initialized to the location in a garden where the Insect is placed when it is constructed. direction, which will be initialized to the direction the Insect is facing when placed in the garden. Which variable in the Insect class should be static?

- **A.** position and direction
- **B.** age and idNum
- **C.** All of them
- **D.** idNum
- **E.** nextAvailableID and position
- **F.** None of them
- **G.** direction
- **H.** position
- **I.** age
- **J.** nextAvailableID

**Answer: J**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_016*

---

## Q17. Explain the action of the following procedure which in-cludes asubroutine procedure within itself. VERIFY_TEST_VALUES: PROCEDURE; DCL LENGTH FIXEDDEC(3); CALL GET_AND_TEST_INPUT; \textbullet \textbullet \textbullet \textbullet \textbullet CALL GET_AND_TEST_INPUT' \textbullet \textbullet \textbullet \textbullet \textbullet GET_AND_TEST_INPUT:PROCEDURE; AGAIN:GETLIST(LENGTH); IF LENGTH = 0 THEN GOTO L; IF LENGTH<0 \vert LENGTH>90 THEN DO; PUTLIST('ERROR', LENGTH);GOTOAGAIN; END; /\textasteriskcentered END OF DO GROUP \textasteriskcentered/ END GET_AND_TEST_INPUT; \textbullet \textbullet \textbullet \textbullet \textbullet CALL GET_AND_TEST_INPUT; L:ENDVERIFY_TEST_VALUES;

- **A.** The subroutine procedure checks if each length value is between 0 and 100
- **B.** The subroutine procedure returns the length value to the main program
- **C.** The subroutine procedure is called only once in the program
- **D.** The subroutine procedure is called at the beginning and end of the main program to validate the length
- **E.** The subroutine procedure is called multiple times, each time incrementing the length value by 1
- **F.** The subroutine procedure is called from three different points in the program, checks if each length value is between 0 and 90, and returns control to the main program.
- **G.** The subroutine procedure is used to exit the main program if the length is within the specified range
- **H.** The subroutine procedure is called recursively within itself to handle multiple length values
- **I.** The subroutine procedure generates a list of length values from 0 to 90
- **J.** The subroutine procedure is a loop that continues until the length value is outside the range of 0 to 90

**Answer: F**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_017*

---

## Q18. Consider Convolutional Neural Network D2 which takes input images of size 32x32 with 1 colour channels. The first layer of D2 uses 4 filters of size 5x5, a stride of 2, and zero-padding of width 1. Consider CNN D2 which takes input images of size 32x32 with 1 colour channels. The first layer of D2 uses 4 filters of size 5x5, a stride of 2, and zero-padding of width 1. What is the total number of weights defined for the entire activation output of this first layer? (ie. If you flattened all filters and channels into a single vector)

- **A.** 1200
- **B.** 500
- **C.** 1300
- **D.** 1040
- **E.** 1100
- **F.** 600
- **G.** 900
- **H.** 1000
- **I.** 700
- **J.** 800

**Answer: G**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_018*

---

## Q19. You have just received a Basic Assembler Language programyou ran. As you turn to the execution page to check theprintout, you find that every fourth line of the program reads, for example, PSW =... 00004F17B4C4. What is the PSW and what information does it provide? In addition, what informationis provided in the three lines following the PSW statement?

- **A.** The three lines following the PSW statement are error codes.
- **B.** PSW is a password for system protection.
- **C.** PSW is a post-system warning that flags potential security breaches, with the subsequent lines listing the affected files.
- **D.** PSW denotes the Previous Software Version, with the following lines outlining the changes made in the current version.
- **E.** PSW refers to Processor Status Word, indicating the processor's current tasks. The lines after detail the task queue.
- **F.** PSW stands for Program Submission Workflow, showing the process flow, while the following lines provide the timestamps of each step completion.
- **G.** PSW means Program Step Width, representing the step size in memory allocation; the next lines show memory usage statistics.
- **H.** PSW is a programming language used by the system.
- **I.** PSW is the identifier for the program start window, and the following lines are user credentials.
- **J.** PSW stands for Program Status Word. It contains the value of the location counter, system protection information, and program interrupt status. The three lines following the PSW statement display the contents of the system's registers at the time of interruption.

**Answer: J**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_019*

---

## Q20. Statement 1| After mapped into feature space Q through a radial basis kernel function, 1-NN using unweighted Euclidean distance may be able to achieve better classification performance than in original space (though we can’t guarantee this). Statement 2| The VC dimension of a Perceptron is smaller than the VC dimension of a simple linear SVM.

- **A.** Partly True, Partly False
- **B.** Partly False, Partly True
- **C.** True, False
- **D.** True, Partly True
- **E.** False, False
- **F.** True, True
- **G.** False, Partly True
- **H.** Partly False, True
- **I.** Partly True, False
- **J.** False, True

**Answer: E**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_020*

---

## Q21. Let a undirected graph G with edges E = {<2,1>,<2,0>,<2,3>,<1,4>,<4,3>}, which <A,B> represent Node A is connected to Node B. What is the minimum vertex cover of G? Represent the vertex cover in a list of ascending order.

- **A.** [2, 3]
- **B.** [0, 2]
- **C.** [0, 1, 2]
- **D.** [1, 3]
- **E.** [0, 1, 3]
- **F.** [1, 2, 4]
- **G.** [2, 4]
- **H.** [1, 2, 3]
- **I.** [0, 2, 3, 4]
- **J.** [2, 3, 4]

**Answer: G**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_021*

---

## Q22. Let a undirected graph G with edges E = {<1,2>,<2,4>,<5,4>,<5,6>}, which <A,B> represent Node A is connected to Node B. What is the shortest path from node 1 to node 6? Represent the path as a list.

- **A.** [1, 5, 4, 6]
- **B.** [1, 5, 4, 2, 6]
- **C.** [1, 2, 4, 5, 6]
- **D.** [1, 2, 5, 4, 6]
- **E.** [1, 2, 5, 6]
- **F.** [1, 2, 4, 5]
- **G.** [1, 2, 5, 4, 5, 6]
- **H.** [1, 5, 6]
- **I.** [1, 2, 6]
- **J.** [1, 2, 4, 6]

**Answer: C**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_022*

---

## Q23. In writing a basic assembler language program, our first and foremost concern is that of housekeeping. What is housekeep-ing and why is it necessary?

- **A.** Housekeeping instructions only load the base register for the program
- **B.** Housekeeping refers to commenting on every line of the code to increase readability
- **C.** Housekeeping involves the deletion of temporary files created during the compilation of the program
- **D.** Housekeeping instructions are used to allocate memory for variables and constants used in the program
- **E.** Housekeeping involves setting the color scheme of the editor for better visibility of the code
- **F.** Housekeeping ensures that the assembler language program is encrypted for security purposes
- **G.** Housekeeping instructions are optional in a program
- **H.** Housekeeping instructions provide standard linkage between the program and the operating system, load and identify the base register for the program, and prepare the units and materials needed by the main processing part of the program.
- **I.** Housekeeping is the process of optimizing the code to run faster on the machine
- **J.** Housekeeping is only concerned with the preparation of units and materials

**Answer: H**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_023*

---

## Q24. The following signal $x_1(t)=\cos (3 \pi t)-4 \cos (5 \pi t-0.5 \pi)$ can be expressed as $x_1(t)=\operatorname{Real}\left(A e^{j \pi B t}\right)+\operatorname{Real}\left(D e^{j \pi E t}\right)$. What are B,E?

- **A.** [2, 5]
- **B.** [5, 3]
- **C.** [7, 4]
- **D.** [1, 6]
- **E.** [5, 2]
- **F.** [3, 5]
- **G.** [2, 4]
- **H.** [6, 1]
- **I.** [3, 7]
- **J.** [4, 7]

**Answer: F**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_024*

---

## Q25. You want to cluster 7 points into 3 clusters using the k-Means Clustering algorithm. Suppose after the first iteration, clusters C1, C2 and C3 contain the following two-dimensional points: C1 contains the 2 points: {(0,6), (6,0)} C2 contains the 3 points: {(2,2), (4,4), (6,6)} C3 contains the 2 points: {(5,5), (7,7)} What are the cluster centers computed for these 3 clusters?

- **A.** C1: (3,3), C2: (6,6), C3: (12,12)
- **B.** C1: (6,6), C2: (12,12), C3: (12,12)
- **C.** C1: (3,3), C2: (4,4), C3: (6,6)
- **D.** C1: (1,1), C2: (4,4), C3: (6,6)
- **E.** C1: (0,0), C2: (4,4), C3: (6,6)
- **F.** C1: (6,0), C2: (4,4), C3: (7,7)
- **G.** C1: (3,3), C2: (2,2), C3: (7,7)
- **H.** C1: (2,2), C2: (5,5), C3: (7,7)
- **I.** C1: (0,0), C2: (48,48), C3: (35,35)
- **J.** C1: (0,6), C2: (2,2), C3: (5,5)

**Answer: C**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_025*

---

## Q26. Statement 1| L2 regularization of linear models tends to make models more sparse than L1 regularization. Statement 2| Residual connections can be found in ResNets and Transformers.

- **A.** False, Mostly False
- **B.** False, True
- **C.** Mostly True, True
- **D.** False, False
- **E.** True, False
- **F.** Mostly False, Mostly False
- **G.** Mostly True, False
- **H.** True, True
- **I.** True, Mostly True
- **J.** Mostly False, True

**Answer: B**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_026*

---

## Q27. What is the output of the following program? main ( ) { intx = 5; inty = 5; printf("%d%d\textbackslashn", x++, x); printf("%d%d\textbackslashn", ++y, y); }

- **A.** 56, 65
- **B.** 65, 56
- **C.** 66, 55
- **D.** 55, 67
- **E.** 55, 66
- **F.** 55, 55
- **G.** 65, 67
- **H.** 67, 55
- **I.** 66, 66
- **J.** 56, 66

**Answer: A**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_027*

---

## Q28. Let a undirected graph G with edges E = {<0,2>, <2,4>, <3,4>, <1,4>}, which <A,B> represent Node A is connected to Node B. What is the minimum vertex cover of G if 0 is one of vertex cover? Represent the vertex cover in a list of ascending order.

- **A.** [0, 1, 2]
- **B.** [0, 2, 4]
- **C.** [0, 1, 4]
- **D.** [0, 4]
- **E.** [0, 3, 4]
- **F.** [0, 1, 3]
- **G.** [0, 1]
- **H.** [0, 2, 3]
- **I.** [0, 2]
- **J.** [0, 3]

**Answer: D**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_028*

---

## Q29. Statement 1| The SVM learning algorithm is guaranteed to find the globally optimal hypothesis with respect to its object function. Statement 2| After being mapped into feature space Q through a radial basis kernel function, a Perceptron may be able to achieve better classification performance than in its original space (though we can’t guarantee this).

- **A.** False, True
- **B.** True, False
- **C.** Statement 1 is False, Statement 2 is not applicable
- **D.** False, False
- **E.** Statement 1 is True, Statement 2 is not applicable
- **F.** True, True
- **G.** Statement 1 is partially True, Statement 2 is False
- **H.** Statement 1 and Statement 2 are both not applicable
- **I.** Statement 1 is not applicable, Statement 2 is False
- **J.** Statement 1 is not applicable, Statement 2 is True

**Answer: F**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_029*

---

## Q30. Suppose you are given an EM algorithm that finds maximum likelihood estimates for a model with latent variables. You are asked to modify the algorithm so that it finds MAP estimates instead. Which step or steps do you need to modify?

- **A.** Only the final iteration
- **B.** Expectation and Maximization
- **C.** The likelihood function only
- **D.** Initialization
- **E.** No modification necessary
- **F.** Maximization
- **G.** Both
- **H.** Convergence Criteria
- **I.** Expectation
- **J.** The algorithm cannot be modified to find MAP estimates

**Answer: F**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_030*

---

## Q31. Statement 1| For a continuous random variable x and its probability distribution function p(x), it holds that 0 ≤ p(x) ≤ 1 for all x. Statement 2| Decision tree is learned by minimizing information gain.

- **A.** Statement 1 is a well-known inequality in calculus, Statement 2 is a common method in data mining
- **B.** Statement 1 is a theorem in real analysis, Statement 2 is a principle in information theory
- **C.** Statement 1 is a fundamental principle of statistics, Statement 2 is a rule of thumb in decision analysis
- **D.** False, True
- **E.** False, False
- **F.** True, True
- **G.** Statement 1 is a fundamental principle of probability, Statement 2 is a common algorithm for machine learning
- **H.** Statement 1 is a key concept in measure theory, Statement 2 is a method used in decision theory
- **I.** Statement 1 is a property of all random variables, Statement 2 is a heuristic used in artificial intelligence
- **J.** True, False

**Answer: E**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_031*

---

## Q32. Company X shipped 5 computer chips, 1 of which was defective, and Company Y shipped 4 computer chips, 2 of which were defective. One computer chip is to be chosen uniformly at random from the 9 chips shipped by the companies. If the chosen chip is found to be defective, what is the probability that the chip came from Company Y?

- **A.** 3 / 9
- **B.** 4 / 9
- **C.** 5 / 9
- **D.** 5 / 6
- **E.** 3 / 4
- **F.** 1 / 3
- **G.** 1 / 2
- **H.** 2 / 9
- **I.** 2 / 3
- **J.** 1 / 9

**Answer: I**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_032*

---

## Q33. We are interested in the capacity of photographic film. The film consists of silver iodide crystals, Poisson distributed, with a density of 100 particles per unit area. The film is illuminated without knowledge of the position of the silver iodide particles. It is then developed and the receiver sees only the silver iodide particles that have been illuminated. It is assumed that light incident on a cell exposes the grain if it is there and otherwise results in a blank response. Silver iodide particles that are not illuminated and vacant portions of the film remain blank. We make the following assumptions: We grid the film very finely into cells of area $dA$. It is assumed that there is at most one silver iodide particle per cell and that no silver iodide particle is intersected by the cell boundaries. Thus, the film can be considered to be a large number of parallel binary asymmetric channels with crossover probability $1 - 100dA$. What is the capacity of a 0.1 unit area film?

- **A.** 100.0
- **B.** 5.0
- **C.** 0.5
- **D.** 50.0
- **E.** 0.1
- **F.** 2.0
- **G.** 20.0
- **H.** 0.01
- **I.** 1.0
- **J.** 10.0

**Answer: J**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_033*

---

## Q34. Given an image $$ \begin{array}{llllllll} 6 & 5 & 6 & 7 & 7 & 7 & 7 & 7 \\ 7 & 7 & 7 & 7 & 6 & 7 & 7 & 7 \\ 8 & 8 & 8 & 6 & 5 & 5 & 6 & 7 \\ 8 & 8 & 8 & 6 & 4 & 3 & 5 & 7 \\ 7 & 8 & 8 & 6 & 3 & 3 & 4 & 6 \\ 7 & 8 & 8 & 6 & 4 & 3 & 4 & 6 \\ 8 & 8 & 8 & 7 & 5 & 5 & 5 & 5 \\ 8 & 9 & 9 & 8 & 7 & 6 & 6 & 4 \end{array} $$ . Find an appropriate threshold for thresholding the following image into 2 regions using the histogram.

- **A.** 8.00
- **B.** 3.50
- **C.** 5.00
- **D.** 4.75
- **E.** 5.75
- **F.** 6.75
- **G.** 4.50
- **H.** 6.25
- **I.** 7.00
- **J.** 7.50

**Answer: H**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_034*

---

## Q35. What is the number of labelled forests on 10 vertices with 5 connected components, such that vertices 1, 2, 3, 4, 5 all belong to different connected components?

- **A.** 70000
- **B.** 50000
- **C.** 30000
- **D.** 55000
- **E.** 75000
- **F.** 60000
- **G.** 45000
- **H.** 100000
- **I.** 80000
- **J.** 40000

**Answer: B**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_035*

---

## Q36. Statement 1| Support vector machines, like logistic regression models, give a probability distribution over the possible labels given an input example. Statement 2| We would expect the support vectors to remain the same in general as we move from a linear kernel to higher order polynomial kernels.

- **A.** True, False
- **B.** False, Not applicable
- **C.** Not applicable, False
- **D.** False, False
- **E.** True, True
- **F.** False, True
- **G.** Not applicable, Not applicable
- **H.** Not applicable, True
- **I.** False, Not specified
- **J.** True, Not applicable

**Answer: D**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_036*

---

## Q37. Statement 1| RoBERTa pretrains on a corpus that is approximate 10x larger than the corpus BERT pretrained on. Statement 2| ResNeXts in 2018 usually used tanh activation functions.

- **A.** False, False
- **B.** False, True
- **C.** True, True
- **D.** Both are partially false
- **E.** Both are partially true
- **F.** Statement 1 is completely true, Statement 2 is partially false
- **G.** Statement 1 is mostly true, Statement 2 is false
- **H.** Statement 1 is completely false, Statement 2 is partially true
- **I.** True, False
- **J.** Statement 1 is false, Statement 2 is mostly true

**Answer: I**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_037*

---

## Q38. The openSSL implementation described in “Remote Timing Attacks are Practical” (by Brumley and Boneh) uses the following performance optimizations: Chinese Remainder (CR), Montgomery Representation (MR), Karatsuba Multiplication (KM), and Repeated squaring and Sliding windows (RS). Which of the following options would close the timing channel attack described in the paper if you turned the listed optimizations off?
1. CR and MR
2. CR

- **A.** KM and RS
- **B.** True, True
- **C.** False, True
- **D.** True, False
- **E.** False, False
- **F.** MR and CR
- **G.** KM, MR, RS, CR
- **H.** RS and MR
- **I.** KM, RS, CR
- **J.** MR, KM, RS

**Answer: B**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_038*

---

## Q39. Let T (n) be defined by T(1) = 7 and T(n + 1) = 3n + T(n) for all integers n ≥ 1. Which of the following represents the order of growth of T(n) as a function of n?

- **A.** Θ(n^3 log n)
- **B.** Θ(n log n^2)
- **C.** Θ(sqrt(n))
- **D.** Θ(n log n)
- **E.** Θ(log n)
- **F.** Θ(n^3)
- **G.** Θ(n)
- **H.** Θ(n^2 sqrt(n))
- **I.** Θ(n^2 log n)
- **J.** Θ(n^2)

**Answer: J**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_039*

---

## Q40. Which of the following is (are) true about virtual memory systems that use pages?
I. The virtual address space can be larger than the amount of physical memory.
II. Programs must be resident in main memory throughout their execution.
III. Pages correspond to semantic characteristics of the program.

- **A.** I and III
- **B.** None of the above
- **C.** All of the above
- **D.** I and II and not III
- **E.** I, II and III
- **F.** III only
- **G.** I only
- **H.** II and III
- **I.** I and II
- **J.** II only

**Answer: G**

*Source: mmlu_pro_computer_science | Category: coding | ID: coding_040*

---

## Q41. Newton's law of cooling states that the temperature of an object changes at a rate proportional to the difference between its temperature and that of its surroundings. Suppose that the temperature of a cup of coffee obeys Newton's law of cooling. If the coffee has a temperature of $200^{\circ} \mathrm{F}$ when freshly poured, and $1 \mathrm{~min}$ later has cooled to $190^{\circ} \mathrm{F}$ in a room at $70^{\circ} \mathrm{F}$, determine when the coffee reaches a temperature of $150^{\circ} \mathrm{F}$.

- **A.** 10.2 min
- **B.** 7.8 min
- **C.** 12.0 min
- **D.** 9.5 min
- **E.** 2.8 min
- **F.** 4.5 min
- **G.** 5.2 min
- **H.** 8.4 min
- **I.** 3.3 min
- **J.** 6.07 min

**Answer: J**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_001*

---

## Q42. A tank originally contains $100 \mathrm{gal}$ of fresh water. Then water containing $\frac{1}{2} \mathrm{lb}$ of salt per gallon is poured into the tank at a rate of $2 \mathrm{gal} / \mathrm{min}$, and the mixture is allowed to leave at the same rate. After $10 \mathrm{~min}$ the process is stopped, and fresh water is poured into the tank at a rate of $2 \mathrm{gal} / \mathrm{min}$, with the mixture again leaving at the same rate. Find the amount of salt in the tank at the end of an additional $10 \mathrm{~min}$.

- **A.** 6.3 lb
- **B.** 7.42 lb
- **C.** 7 lb
- **D.** 10 lb
- **E.** 8.5 lb
- **F.** 9.5 lb
- **G.** 9 lb
- **H.** 6 lb
- **I.** 5 lb
- **J.** 8.1 lb

**Answer: B**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_002*

---

## Q43. Mr. Cleary’s class and Ms. Ntuala’s class go to use the computer lab. There are 20 computers available, two of which do not work. Mr. Cleary’s class has 14 kids, and Ms. Ntuala’s class has 12 kids. If every student must use a computer and there can only be 2 students on a computer at most, what is the maximum number of students who can have a computer to themselves?

- **A.** 2
- **B.** 18
- **C.** 8
- **D.** 6
- **E.** 16
- **F.** 12
- **G.** 20
- **H.** 10
- **I.** 24
- **J.** 14

**Answer: H**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_003*

---

## Q44. The president of an online music streaming service whose customers pay a fee wants to gather additional information about customers who have joined in the past 12 months. The company plans to send out an e-mail survey to a sample of current customers with a link that gives participants a month of streaming service for free once the survey has been completed. They know that musical tastes vary by geographical region. Which of the following sample plans would produce the most representative sample of its customers?

- **A.** Choose all of the customers who joined in the last 6 months.
- **B.** From the list of all customers who joined in the last 12 months, classify customers by the state in which they live, then choose 10 customers from each state.
- **C.** Choose all of the customers who joined in the last month.
- **D.** Choose a random sample of customers who joined in the last 12 months and have streamed at least 100 songs.
- **E.** Make a list of all the customers who joined in the last 12 months and choose a random sample of customers on this list.
- **F.** From the list of all customers who joined in the last 12 months, classify customers by the city in which they live, then choose 5 customers from each city.
- **G.** Choose all of the customers who have joined in the last 12 months and have streamed at least 200 songs.
- **H.** From the list of all customers who joined in the last 12 months, classify customers by the country in which they live, then choose 5% of the customers from each country.
- **I.** From the list of all customers who joined in the last 12 months, classify customers by the genre of music they most frequently listen to, then choose a random sample from each genre.
- **J.** From the list of all customers who joined in the last 12 months, classify customers by the state in which they live, then choose 3% of the customers from each state.

**Answer: J**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_004*

---

## Q45. A researcher planning a survey of school principals in a particular state has lists of the school principals employed in each of the 125 school districts. The procedure is to obtain a random sample of principals from each of the districts rather than grouping all the lists together and obtaining a sample from the entire group. Which of the following is a correct conclusion?

- **A.** This is a purposive sampling, where the researcher only selects individuals that meet specific characteristics.
- **B.** This is a non-probability sampling, where the samples are gathered in a process that does not give all the individuals in the population equal chances of being selected.
- **C.** This is a stratified sample, which may give comparative information that a simple random sample wouldn't give.
- **D.** This is a cluster sample in which the population was divided into heterogeneous groups called clusters.
- **E.** This is a convenience sampling, where the researcher chooses the easiest group to reach.
- **F.** This is a snowball sampling, where existing study subjects recruit future subjects.
- **G.** This is an example of systematic sampling, which gives a reasonable sample as long as the original order of the list is not related to the variables under consideration.
- **H.** This is a quota sampling, where the researcher ensures equal representation of each district.
- **I.** This is a sequential sampling, where sampling continues until a certain criteria is met.
- **J.** This is a simple random sample obtained in an easier and less costly manner than procedures involving sampling from the entire population of principals.

**Answer: C**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_005*

---

## Q46. A college admissions officer is interested in comparing the SAT math scores of high school applicants who have and have not taken AP Statistics. She randomly pulls the files of five applicants who took AP Statistics and five applicants who did not, and proceeds to run a t-test to compare the mean SAT math scores of the two groups. Which of the following is a necessary assumption?

- **A.** The population of SAT scores from each group is not normally distributed.
- **B.** The SAT scores from each group are unrelated.
- **C.** The mean SAT scores from each group are similar.
- **D.** The SAT scores from each group are skewed.
- **E.** The population variances from each group are known.
- **F.** The population of SAT scores from each group is normally distributed.
- **G.** The population variances from the two groups are unequal.
- **H.** The population variances from each group are unknown.
- **I.** The population variances from the two groups are equal.
- **J.** The population variances from each group are estimated.

**Answer: F**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_006*

---

## Q47. A soft drink dispenser can be adjusted to deliver any fixed number of ounces. If the machine is operating with a standard deviation in delivery equal to 0.3 ounce, what should be the mean setting so that a 12-ounce cup will overflow less than 1% of the time? Assume a normal distribution for ounces delivered.

- **A.** 12 + 0.99(0.3) ounces
- **B.** 12 - 1.645(0.3) ounces
- **C.** 12 + 2.326(0.3) ounces
- **D.** 12 - 1.96(0.3) ounces
- **E.** 12 + 2.576(0.3) ounces
- **F.** 12 + 1.645(0.3) ounces
- **G.** 12 - 0.99(0.3) ounces
- **H.** 12 - 2.576(0.3) ounces
- **I.** 12 + 1.96(0.3) ounces
- **J.** 12 - 2.326(0.3) ounces

**Answer: J**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_007*

---

## Q48. The Department of Health plans to test the lead level in a specific park. Because a high lead level is harmful to children, the park will be closed if the lead level exceeds the allowed limit. The department randomly selects several locations in the park, gets soil samples from those locations, and tests the samples for their lead levels. Which of the following decisions would result from the type I error?

- **A.** Keeping the park open when the lead levels are within the allowed limit
- **B.** Closing the park based on a predicted future lead level increase
- **C.** Keeping the park open when the lead levels are in excess of the allowed limit
- **D.** Closing other parks when their lead levels are within the allowed limit
- **E.** Closing the park when the lead levels are in excess of the allowed limit
- **F.** Not testing the lead levels at all and closing the park
- **G.** Keeping other parks open when their lead levels are in excess of the allowed limit
- **H.** Disregarding the test results and keeping the park open regardless of the lead levels
- **I.** Closing the park when the lead levels are within the allowed limit
- **J.** Closing the park based on lead levels from a different park

**Answer: I**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_008*

---

## Q49.  A typical roulette wheel used in a casino has 38 slots that are numbered $1,2,3, \ldots, 36,0,00$, respectively. The 0 and 00 slots are colored green. Half of the remaining slots are red and half are black. Also, half of the integers between 1 and 36 inclusive are odd, half are even, and 0 and 00 are defined to be neither odd nor even. A ball is rolled around the wheel and ends up in one of the slots; we assume that each slot has equal probability of $1 / 38$, and we are interested in the number of the slot into which the ball falls. Let $A=\{0,00\}$. Give the value of $P(A)$.

- **A.** $\frac{1}{19}$
- **B.** $\frac{1}{76}$
- **C.** $\frac{20}{38}$
- **D.** $\frac{2}{38}$
- **E.** $\frac{3}{38}$
- **F.** $\frac{4}{38}$
- **G.** $\frac{36}{38}$
- **H.** $\frac{1}{38}$
- **I.** $\frac{1}{2}$
- **J.** $\frac{18}{38}$

**Answer: D**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_009*

---

## Q50. A mass of $100 \mathrm{~g}$ stretches a spring $5 \mathrm{~cm}$. If the mass is set in motion from its equilibrium position with a downward velocity of $10 \mathrm{~cm} / \mathrm{s}$, and if there is no damping, determine when does the mass first return to its equilibrium position.

- **A.** $3\pi/14$ s
- **B.** $2\pi/7$ s
- **C.** $\pi/5$ s
- **D.** $\pi/10$ s
- **E.** $\pi/20$ s
- **F.** $\pi/6$ s
- **G.** $\pi/12$ s
- **H.** $\pi/8$ s
- **I.** $\pi/7$ s
- **J.** $\pi/14$ s

**Answer: J**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_010*

---

## Q51. 7.4-5. A quality engineer wanted to be $98 \%$ confident that the maximum error of the estimate of the mean strength, $\mu$, of the left hinge on a vanity cover molded by a machine is 0.25 . A preliminary sample of size $n=32$ parts yielded a sample mean of $\bar{x}=35.68$ and a standard deviation of $s=1.723$.
(a) How large a sample is required?

- **A.** $275$
- **B.** $210$
- **C.** $190$
- **D.** $170$
- **E.** $245$
- **F.** $257$
- **G.** $320$
- **H.** $225$
- **I.** $300$
- **J.** $150$

**Answer: F**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_011*

---

## Q52. A bin has 5 green balls and $k$ purple balls in it, where $k$ is an unknown positive integer. A ball is drawn at random from the bin. If a green ball is drawn, the player wins 2 dollars, but if a purple ball is drawn, the player loses 2 dollars. If the expected amount won for playing the game is 50 cents, then what is $k$?

- **A.** 5
- **B.** 7
- **C.** 12
- **D.** 3
- **E.** 10
- **F.** 4
- **G.** 9
- **H.** 2
- **I.** 6
- **J.** 8

**Answer: D**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_012*

---

## Q53. A drug company will conduct a randomized controlled study on the effectiveness of a new heart disease medication called Heartaid. Heartaid is more expensive than the currently used medication. The analysis will include a significance test with H0: Heartaid and the current medication are equally effective at preventing heart disease and HA: Heartaid is more effective than the current medication at preventing heart disease. Which of these would be a potential consequence of a Type II error?

- **A.** Patients will suffer from side effects of Heartaid, even though it is not any more effective than the current medication.
- **B.** The drug company will continue to produce Heartaid, even though it is not any more effective than the current medication.
- **C.** Doctors will stop prescribing the current medication, even though Heartaid is not any more effective.
- **D.** Researchers will incorrectly reject the null hypothesis, leading to a false conclusion about the effectiveness of Heartaid.
- **E.** Researchers will calculate the wrong P-value, making their advice to doctors invalid.
- **F.** The drug company will lose money because Heartaid is actually not any more effective than the current medication.
- **G.** Patients will continue to use the current medication, even though Heartaid is actually more effective.
- **H.** Doctors will begin to prescribe Heartaid to patients, even though it is actually not any more effective than the current medication.
- **I.** The FDA will approve Heartaid for use, even though it is not any more effective than the current medication.
- **J.** Patients will spend more money on Heartaid, even though it is actually not any more effective than the current medication.

**Answer: G**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_013*

---

## Q54. One model for the spread of an epidemic is that the rate of spread is jointly proportional to the number of infected people and the number of uninfected people. In an isolated town of 5000 inhabitants, 160 people have a disease at the beginning of the week and 1200 have it at the end of the week. How long does it take for $80 \%$ of the population to become infected?

- **A.** 22 days
- **B.** 20 $\mathrm{days}$
- **C.** 12 days
- **D.** 18 days
- **E.** 8 days
- **F.** 30 days
- **G.** 25 $\mathrm{days}$
- **H.** 10 $\mathrm{days}$
- **I.** 15 $\mathrm{days}$
- **J.** 35 days

**Answer: I**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_014*

---

## Q55. To determine the average number of children living in single-family homes, a researcher picks a simple random sample of 50 such homes. However, even after one follow-up visit the interviewer is unable to make contact with anyone in 8 of these homes. Concerned about nonresponse bias, the researcher picks another simple random sample and instructs the interviewer to keep trying until contact is made with someone in a total of 50 homes. The average number of children is determined to be 1.73. Is this estimate probably too low or too high?

- **A.** Too high, because the sample size is not large enough.
- **B.** Too low, because nonresponse bias tends to underestimate average results.
- **C.** Too high, because convenience samples overestimate average results.
- **D.** Too low, because of undercoverage bias.
- **E.** Too low, because the sample size is not large enough.
- **F.** Too high, due to the possibility of a skewed sample.
- **G.** Too high, because of undercoverage bias.
- **H.** Too low, because convenience samples overestimate average results.
- **I.** Too low, due to the possibility of a skewed sample.
- **J.** Too high, because nonresponse bias tends to overestimate average results.

**Answer: J**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_015*

---

## Q56. The total cholesterol level in a large population of people is strongly skewed right with a mean of 210 mg/dL and a standard deviation of 15 mg/dL. If random samples of size 16 are repeatedly drawn from this population, which of the following appropriately describes the sampling distribution of these sample means?

- **A.** The shape is somewhat skewed left with a mean of 210 and a standard deviation of 3.75.
- **B.** The shape is strongly skewed left with a mean of 210 and a standard deviation of 3.75.
- **C.** The shape is approximately normal with a mean of 210 and a standard deviation of 3.75.
- **D.** The shape is approximately normal with a mean of 200 and a standard deviation of 3.75.
- **E.** The shape is strongly skewed right with a mean of 225 and a standard deviation of 3.75.
- **F.** The shape is approximately normal with a mean of 210 and a standard deviation of 15.
- **G.** The shape is unknown with a mean of 210 and a standard deviation of 15.
- **H.** The shape is approximately normal with a mean of 210 and a standard deviation of 7.5.
- **I.** The shape is somewhat skewed right with a mean of 210 and a standard deviation of 3.75.
- **J.** The shape is strongly skewed right with a mean of 210 and a standard deviation of 15.

**Answer: I**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_016*

---

## Q57. Let R be a ring with a multiplicative identity. If U is an additive subgroup of R such that ur in U for all u in U and for all r in R, then U is said to be a right ideal of R. If R has exactly two right ideals, which of the following must be true?
I. R is commutative.
II. R is a division ring (that is, all elements except the additive identity have multiplicative inverses).
III. R is infinite.

- **A.** R is a finite ring
- **B.** R is a ring without a multiplicative identity
- **C.** II and III only
- **D.** II only
- **E.** I and III only
- **F.** I and II only
- **G.** None of the above
- **H.** III only
- **I.** I only
- **J.** I, II and III

**Answer: D**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_017*

---

## Q58. suppose a,b,c,\alpha,\beta,\gamma are six real numbers with a^2+b^2+c^2>0.  In addition, $a=b*cos(\gamma)+c*cos(\beta), b=c*cos(\alpha)+a*cos(\gamma), c=a*cos(\beta)+b*cos(\alpha)$. What is the value of $cos^2(\alpha)+cos^2(\beta)+cos^2(\gamma)+2*cos(\alpha)*cos(\beta)*cos(\gamma)? return the numeric.

- **A.** 0.5
- **B.** 0.0
- **C.** 2.0
- **D.** 1.0
- **E.** -0.5
- **F.** 1.5
- **G.** -1.0
- **H.** 2.5
- **I.** 3.0
- **J.** 0.25

**Answer: D**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_018*

---

## Q59. A six-sided die (whose faces are numbered 1 through 6, as usual) is known to be counterfeit: The probability of rolling any even number is twice the probability of rolling any odd number. What is the probability that if this die is thrown twice, the first roll will be a 5 and the second roll will be a 6?

- **A.** 1/27
- **B.** 2/81
- **C.** 1/9
- **D.** 1/18
- **E.** 2/9
- **F.** 3/81
- **G.** 2/27
- **H.** 1/81
- **I.** 3/27
- **J.** 2/18

**Answer: B**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_019*

---

## Q60. An urn contains four balls numbered 1 through 4 . The balls are selected one at a time without replacement. A match occurs if the ball numbered $m$ is the $m$ th ball selected. Let the event $A_i$ denote a match on the $i$ th draw, $i=1,2,3,4$. Extend this exercise so that there are $n$ balls in the urn. What is the limit of this probability as $n$ increases without bound?

- **A.** $1 - \frac{1}{e}$
- **B.** $\frac{1}{e}$
- **C.** $\frac{1}{n!}$
- **D.** $\frac{1}{\sqrt{n}}$
- **E.** $\frac{1}{4}$
- **F.** $\frac{2}{e}$
- **G.** $\frac{1}{2}$
- **H.** $e^{-1/n}$
- **I.** $\frac{n-1}{n}$
- **J.** $\frac{1}{n}$

**Answer: A**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_020*

---

## Q61. The lowest point on Earth is the bottom of the Mariana Trench at a depth of 35,840 feet below sea level. The highest point on Earth is the summit of Mt. Everest at a height of 29,028 feet above sea level. Which of the following is the best estimate of the distance between the lowest and highest points on Earth?

- **A.** 75,000 feet
- **B.** 65,000 feet
- **C.** 55,000 feet
- **D.** 60,000 feet
- **E.** 7,000 feet
- **F.** 64,000 feet
- **G.** 70,000 feet
- **H.** 50,000 feet
- **I.** 6,000 feet
- **J.** 80,000 feet

**Answer: B**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_021*

---

## Q62. You have a coin and you would like to check whether it is fair or biased. More specifically, let $\theta$ be the probability of heads, $\theta = P(H)$. Suppose that you need to choose between the following hypotheses: H_0 (null hypothesis): The coin is fair, i.e. $\theta = \theta_0 = 1 / 2$. H_1 (the alternative hypothesis): The coin is not fair, i.e. $\theta > 1 / 2$. We toss 100 times and observe 60 heads. What is the P-value?

- **A.** 0.115
- **B.** 0.28
- **C.** 0.023
- **D.** 0.157
- **E.** 0.95
- **F.** 0.001
- **G.** 0.5
- **H.** 0.1
- **I.** 0.05
- **J.** 0.082

**Answer: C**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_022*

---

## Q63. 5.4-17. In a study concerning a new treatment of a certain disease, two groups of 25 participants in each were followed for five years. Those in one group took the old treatment and those in the other took the new treatment. The theoretical dropout rate for an individual was $50 \%$ in both groups over that 5 -year period. Let $X$ be the number that dropped out in the first group and $Y$ the number in the second group. Assuming independence where needed, give the sum that equals the probability that $Y \geq X+2$. HINT: What is the distribution of $Y-X+25$ ?


- **A.** $0.1745$
- **B.** $0.3751$
- **C.** $0.5012$
- **D.** $0.3359$
- **E.** $0.3926$
- **F.** $0.2872$
- **G.** $0.6118$
- **H.** $0.4256$
- **I.** $0.4583$
- **J.** $0.2197$

**Answer: D**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_023*

---

## Q64. The amount of paint that David needs to cover a cube is directly proportional to the surface area. If David can completely cover a cube of side length 2 feet with exactly 16 quarts of paint, how big a cube (in terms of edge length in feet) can David cover with 169 quarts of paint?

- **A.** \frac{13}{4}
- **B.** \frac{13}{2}
- **C.** 16
- **D.** 26
- **E.** 13
- **F.** \frac{13}{8}
- **G.** 8
- **H.** \frac{26}{3}
- **I.** \frac{8}{13}
- **J.** 169

**Answer: B**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_024*

---

## Q65. 7.3-3. Let $p$ equal the proportion of triathletes who suffered a training-related overuse injury during the past year. Out of 330 triathletes who responded to a survey, 167 indicated that they had suffered such an injury during the past year.
(a) Use these data to give a point estimate of $p$.

- **A.** $0.5061$
- **B.** $0.6000$
- **C.** $0.4000$
- **D.** $0.7000$
- **E.** $0.5894$
- **F.** $0.4562$
- **G.** $0.6531$
- **H.** $0.3077$
- **I.** $0.7500$
- **J.** $0.5500$

**Answer: A**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_025*

---

## Q66. A manufacturer of ready-bake cake mixes is interested in designing an experiment to test the effects of four different temperature levels (300, 325, 350, and 375F), two different types of pans (glass and metal), and three different types of ovens (gas, electric, and microwave) on the texture of its cakes, in all combinations. Which of the following below is the best description of the design of the necessary experiment?

- **A.** A randomized block design, blocked on temperature and type of pan, with 12 treatment groups
- **B.** A completely randomized design with 6 treatment groups
- **C.** A randomized block design, blocked on type of oven, with 24 treatment groups
- **D.** A randomized block design, blocked on temperature, with six treatment groups
- **E.** A completely randomized design with 18 treatment groups
- **F.** A randomized block design, blocked on type of oven, with 12 treatment groups
- **G.** A completely randomized design with 24 treatment groups
- **H.** A randomized block design, blocked on type of pan, with 12 treatment groups
- **I.** A completely randomized design with 12 treatment groups
- **J.** A completely randomized design with nine treatment groups

**Answer: G**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_026*

---

## Q67. In the casino game of roulette, there are 38 slots for a ball to drop into when it is rolled around the rim of a revolving wheel: 18 red, 18 black, and 2 green. What is the probability that the first time a ball drops into the red slot is on the 8th trial (in other words, suppose you are betting on red every time-what is the probability of losing 7 straight times before you win the first time)?

- **A.** 0.0333
- **B.** 0.0256
- **C.** 0.0202
- **D.** 0.0053
- **E.** 0.0101
- **F.** 0.0112
- **G.** 0.0179
- **H.** 0.0278
- **I.** 0.0074
- **J.** 0.0158

**Answer: D**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_027*

---

## Q68. A shipment of resistors have an average resistance of 200 ohms with a standard deviation of 5 ohms, and the resistances are normally distributed. Suppose a randomly chosen resistor has a resistance under 194 ohms. What is the probability that its resistance is greater than 188 ohms?

- **A.** 0.93
- **B.** 0.07
- **C.** 0.20
- **D.** 0.30
- **E.** 0.35
- **F.** 0.12
- **G.** 0.99
- **H.** 0.65
- **I.** 0.50
- **J.** 0.80

**Answer: A**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_028*

---

## Q69. Anton has two species of ants, Species A and Species B, in his ant farm. The two species are identical in appearance, but Anton knows that every day, there are twice as many ants of Species A than before, while there are three times as many ants of Species B. On Day 0, Anton counts that there are 30 ants in his ant farm. On Day 5, Anton counts that there are 3281 ants in his ant farm. How many of these are of Species A?

- **A.** 608
- **B.** 960
- **C.** 2321
- **D.** 211
- **E.** 500
- **F.** 512
- **G.** 11
- **H.** 1000
- **I.** 2048
- **J.** 150

**Answer: A**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_029*

---

## Q70. 7.3-9. Consider the following two groups of women: Group 1 consists of women who spend less than $\$ 500$ annually on clothes; Group 2 comprises women who spend over $\$ 1000$ annually on clothes. Let $p_1$ and $p_2$ equal the proportions of women in these two groups, respectively, who believe that clothes are too expensive. If 1009 out of a random sample of 1230 women from group 1 and 207 out of a random sample 340 from group 2 believe that clothes are too expensive,
(a) Give a point estimate of $p_1-p_2$.


- **A.** $0.2115$
- **B.** $0.3256$
- **C.** $0.1298$
- **D.** $0.2732$
- **E.** $0.3021$
- **F.** $0.0997$
- **G.** $0.2389$
- **H.** $0.2654$
- **I.** $0.1874$
- **J.** $0.1543$

**Answer: A**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_030*

---

## Q71. Bob rolls a fair six-sided die each morning. If Bob rolls a composite number, he eats sweetened cereal. If he rolls a prime number, he eats unsweetened cereal. If he rolls a 1, then he rolls again. In a non-leap year, what is the expected number of times Bob will roll his die?

- **A.** \frac{365}{4}
- **B.** 365
- **C.** \frac{3}{2}
- **D.** \frac{365}{2}
- **E.** 438
- **F.** \frac{5}{4}
- **G.** 1095
- **H.** 730
- **I.** \frac{5}{8}
- **J.** \frac{1825}{4}

**Answer: E**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_031*

---

## Q72. John is playing a game in which he tries to obtain the highest number possible. He must put the symbols +, $\times$, and - (plus, times, and minus) in the following blanks, using each symbol exactly once:\[2 \underline{\hphantom{8}} 4 \underline{\hphantom{8}} 6 \underline{\hphantom{8}} 8.\] John cannot use parentheses or rearrange the numbers. What is the highest possible number that John could obtain?

- **A.** 22
- **B.** 90
- **C.** 100
- **D.** 78
- **E.** 99
- **F.** 46
- **G.** 56
- **H.** 50
- **I.** 66
- **J.** 38

**Answer: F**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_032*

---

## Q73.  Of a group of patients having injuries, $28 \%$ visit both a physical therapist and a chiropractor and $8 \%$ visit neither. Say that the probability of visiting a physical therapist exceeds the probability of visiting a chiropractor by $16 \%$. What is the probability of a randomly selected person from this group visiting a physical therapist?


- **A.** 0.48
- **B.** 0.52
- **C.** 0.72
- **D.** 0.80
- **E.** 0.60
- **F.** 0.84
- **G.** 0.56
- **H.** 0.44
- **I.** 0.76
- **J.** 0.68

**Answer: J**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_033*

---

## Q74. A scientist studied the migration patterns of two types of whales. The humpback whales traveled 2,240 miles in 28 days. The gray whales traveled 2,368 miles in 32 days. If the humpback whales had traveled at the same rate for 32 days, how many more miles would they have traveled than the gray whales?

- **A.** 64
- **B.** 408
- **C.** 128
- **D.** 320
- **E.** 280
- **F.** 96
- **G.** 192
- **H.** 160
- **I.** 256
- **J.** 224

**Answer: G**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_034*

---

## Q75. A virus is spreading throughout the population of a town, and the number of people who have the virus doubles every 3 days. If there are 1,000 people in the town, and 10 people have the virus on January 1st, what is the earliest date at which the entire town would be infected with the virus, given that there are 365 days in a year, and 31 days in the month of January?

- **A.** February 10th
- **B.** January 28th
- **C.** January 10th
- **D.** March 1st
- **E.** January 15th
- **F.** January 31st
- **G.** March 10th
- **H.** January 21st
- **I.** February 21st
- **J.** February 1st

**Answer: H**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_035*

---

## Q76. Semir rolls a six-sided die every morning to determine what he will have for breakfast. If he rolls a 1 or 2, he takes time to cook himself a big breakfast. If he rolls a 3 or larger he grabs a quick lighter breakfast. When he cooks himself a big breakfast, there is a 15% chance he will be late for school. If he has a lighter breakfast, there is a 6% chance he will be late for school. What is the probability Semir will be on time for school any given day?

- **A.** 0.85
- **B.** 0.1
- **C.** 0.5
- **D.** 0.99
- **E.** 0.91
- **F.** 0.94
- **G.** 0.09
- **H.** 0.75
- **I.** 0.8
- **J.** 0.21

**Answer: E**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_036*

---

## Q77. Water drips out of a hole at the vertex of an upside down cone at a rate of 3 cm^3 per minute. The
cone’s height and radius are 2 cm and 1 cm, respectively. At what rate does the height of the water change
when the water level is half a centimeter below the top of the cone? The volume of a cone is V = (π/3)*r^2*h,
where r is the radius and h is the height of the cone.

- **A.** −28/(3π) cm/min
- **B.** −8/(3π) cm/min
- **C.** −24/(3π) cm/min
- **D.** −48/π cm/min
- **E.** −12/(3π) cm/min
- **F.** −4/(3π) cm/min
- **G.** −32/(3π) cm/min
- **H.** −20/(3π) cm/min
- **I.** −40/(3π) cm/min
- **J.** −16/(3π) cm/min

**Answer: J**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_037*

---

## Q78.  If $R$ is the total resistance of three resistors, connected in parallel, with resistances $R_1, R_2, R_3$, then
$$
\frac{1}{R}=\frac{1}{R_1}+\frac{1}{R_2}+\frac{1}{R_3}
$$
If the resistances are measured in ohms as $R_1=25 \Omega$, $R_2=40 \Omega$, and $R_3=50 \Omega$, with a possible error of $0.5 \%$ in each case, estimate the maximum error in the calculated value of $R$.

- **A.** $\frac{1}{20}$ $\Omega$
- **B.** $\frac{1}{10}$ $\Omega$
- **C.** $\frac{1}{14}$ $\Omega$
- **D.** $\frac{1}{25}$ $\Omega$
- **E.** $\frac{1}{15}$ $\Omega$
- **F.** $\frac{1}{12}$ $\Omega$
- **G.** $\frac{1}{22}$ $\Omega$
- **H.** $\frac{1}{17}$ $\Omega$
- **I.** $\frac{1}{30}$ $\Omega$
- **J.** $\frac{1}{18}$ $\Omega$

**Answer: H**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_038*

---

## Q79. Bowl $B_1$ contains two white chips, bowl $B_2$ contains two red chips, bowl $B_3$ contains two white and two red chips, and bowl $B_4$ contains three white chips and one red chip. The probabilities of selecting bowl $B_1, B_2, B_3$, or $B_4$ are $1 / 2,1 / 4,1 / 8$, and $1 / 8$, respectively. A bowl is selected using these probabilities and a chip is then drawn at random. Find $P(W)$, the probability of drawing a white chip.

- **A.** $\frac{19}{32}$
- **B.** $\frac{13}{32}$
- **C.** $\frac{29}{32}$
- **D.** $\frac{15}{32}$
- **E.** $\frac{27}{32}$
- **F.** $\frac{25}{32}$
- **G.** $\frac{23}{32}$
- **H.** $\frac{17}{32}$
- **I.** $\frac{31}{32}$
- **J.** $\frac{21}{32}$

**Answer: J**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_039*

---

## Q80. A publisher used standard boxes for shipping books. The mean weight of books packed per box is 25 pounds, with a standard deviation of two pounds. The mean weight of the boxes is one pound, with a standard deviation of 0.15 pounds. The mean weight of the packing material used per box is two pounds, with a standard deviation of 0.25 pounds. What is the standard deviation of the weights of the packed boxes?

- **A.** 28.000 pounds
- **B.** 1.950 pounds
- **C.** 2.500 pounds
- **D.** 4.085 pounds
- **E.** 3.012 pounds
- **F.** 3.500 pounds
- **G.** 5.290 pounds
- **H.** 2.021 pounds
- **I.** 2.250 pounds
- **J.** 1.785 pounds

**Answer: H**

*Source: mmlu_pro_math | Category: math_reasoning | ID: math_reasoning_040*

---

## Q81. At steady state conditions, Oxygen (A) diffuses through Nitrogen (B).Consider the nitrogen asnondiffusing. The temperature is 0°C and the total pressure is 1 × 10^5 N / m^2.The partial pressure of oxygen at two locations, 2.2 mm apart is 13500 and 6000 N / m^2. The diffusivityis 1.81 × 10^-5 m^2 / sec. (I)Determinethe diffusion rate of oxygen. (II) Determine the diffusion rate of oxygen (A) using as the nondiffusing gas a mixture of nitrogen (B) and carbon dioxide(C) in a volume ratio of 2:1. The diffusivities are D_(_O)2- (N)2= 1.81 × 10^-5 m^2 / sec, D_(_O)2- (CO)2= 1.85 × 10^-5 m^2 / sec.

- **A.** 2.40 × 10^-5 kmol/m^2-sec, 2.45 × 10^-5 kmol/m^2-sec
- **B.** 2.75 × 10^-5 kmol/m^2-sec, 2.80 × 10^-5 kmol/m^2-sec
- **C.** 2.50 × 10^-5 kmol/m^2-sec, 2.55 × 10^-5 kmol/m^2-sec
- **D.** 2.55 × 10^-5kmol/ m^2-sec, 2.65 × 10^-5kmol/ m^2-sec
- **E.** 2.80 × 10^-5 kmol/m^2-sec, 2.85 × 10^-5 kmol/m^2-sec
- **F.** 2.70 × 10^-5kmol/ m^2-sec, 2.71 × 10^-5kmol/ m^2-sec
- **G.** 2.90 × 10^-5 kmol/m^2-sec, 2.95 × 10^-5 kmol/m^2-sec
- **H.** 2.75 × 10^-5kmol/ m^2-sec, 2.76 × 10^-5kmol/ m^2-sec
- **I.** 2.65 × 10^-5 kmol/m^2-sec, 2.70 × 10^-5 kmol/m^2-sec
- **J.** 2.60 × 10^-5kmol/ m^2-sec, 2.61 × 10^-5kmol/ m^2-sec

**Answer: J**

*Source: mmlu_pro_engineering | Category: science_knowledge | ID: science_knowledge_001*

---

## Q82. Assume that all gases are perfect and that data refer to 298.15 K unless otherwise stated. Consider a system consisting of $2.0 \mathrm{~mol} \mathrm{CO}_2(\mathrm{~g})$, initially at $25^{\circ} \mathrm{C}$ and $10 \mathrm{~atm}$ and confined to a cylinder of cross-section $10.0 \mathrm{~cm}^2$. It is allowed to expand adiabatically against an external pressure of 1.0 atm until the piston has moved outwards through $20 \mathrm{~cm}$. Assume that carbon dioxide may be considered a perfect gas with $C_{V, \mathrm{~m}}=28.8 \mathrm{~J} \mathrm{~K}^{-1} \mathrm{~mol}^{-1}$ and calculate $\Delta U$.

- **A.** -30$\text{J}$
- **B.** -60 J
- **C.** 10 J
- **D.** -40$\text{J}$
- **E.** 0 J
- **F.** -20$\text{J}$ 
- **G.** 30 J
- **H.** -70 J
- **I.** -50 J
- **J.** -10$\text{J}$

**Answer: F**

*Source: mmlu_pro_chemistry | Category: science_knowledge | ID: science_knowledge_002*

---

## Q83. A 30-year-old nulliparous female presents to the office with the complaint of mood changes. She says that for the past several months she has been anxious, hyperactive, and unable to sleep 3 to 4 days prior to the onset of menses. She further reports that on the day her menses begins she becomes acutely depressed, anorectic, irritable, and lethargic. She has no psychiatric history. Physical examination findings are normal. She and her husband have been trying to conceive for over 2 years. History reveals a tuboplasty approximately 1 year ago to correct a closed fallopian tube. The most likely diagnosis is

- **A.** generalized anxiety disorder
- **B.** bipolar I disorder, mixed
- **C.** Premenstrual dysphoric disorder
- **D.** cyclothymic personality
- **E.** Persistent depressive disorder (dysthymia)
- **F.** Post-traumatic stress disorder
- **G.** Seasonal affective disorder
- **H.** Bipolar II disorder
- **I.** adjustment disorder with depressed mood
- **J.** Major depressive disorder

**Answer: I**

*Source: mmlu_pro_health | Category: science_knowledge | ID: science_knowledge_003*

---

## Q84. A male neonate, who was born at 36 weeks' gestation 2 hours ago in the labor and delivery unit of the hospital, now shows signs of respiratory difficulty. The mother, an 18-year-old primigravid woman, smoked one pack of cigarettes daily throughout her pregnancy. She received prenatal care during most of the pregnancy. One episode of chlamydial cervicitis was detected during the last trimester and treated with azithromycin. The neonate was born via cesarean delivery due to fetal heart rate decelerations. The amniotic fluid was stained with light particulate meconium. Apgar scores were 9 and 9 at 1 and 5 minutes, respectively. The patient is 50 cm (20 in; 50th percentile) long and weighs 3005 g (6 lb 10 oz; 50th percentile); head circumference is 35 cm (14 in; 50th percentile). The infant's vital signs now are temperature 36.6°C (97.8°F), pulse 150/min, and respirations 70/min. Pulse oximetry on room air shows an oxygen saturation of 95%. Physical examination discloses mild subcostal and intercostal retractions. Chest x-ray shows prominent pulmonary vascular markings and fluid in the intralobar fissures. Which of the following is the most likely diagnosis? 

- **A.** Group B streptococcal sepsis
- **B.** Pulmonary embolism
- **C.** Respiratory distress syndrome
- **D.** Congenital heart disease
- **E.** Neonatal pneumonia
- **F.** Meconium aspiration syndrome
- **G.** Transient tachypnea of newborn
- **H.** Pulmonary hypoplasia
- **I.** Pneumothorax
- **J.** Pneumonia

**Answer: G**

*Source: mmlu_pro_health | Category: science_knowledge | ID: science_knowledge_004*

---

## Q85. A common method for commercially peeling potatoes is to soakthem for 1 - 5 minutes in a 10 - 20% solution ofNaOH (molecularweight = 40.0 g/mole) at 60 - 88°C, and to spray offthe peel once the potatoes are removed from solution. As aneconomy measure, a manufacturer titrates theNaOH solutionwith standardized H_2SO_4 (molecular weight = 98.0 g/mole) at the end of each day to determine whether the solutionis still capable of peeling potatoes. If, at the end of oneday, he finds that it takes 64.0 ml of a 0.200 M solution ofH_2SO_4 to titrate a 10.0 ml sample ofNaOHsolution to neutrality, what concentration ofNaOHdid he find?

- **A.** 2.56 M
- **B.** 0.32 M
- **C.** 0.16 M
- **D.** 1.28 M
- **E.** 5.12 M
- **F.** 2.00 M
- **G.** 4.00 M
- **H.** 3.12 M
- **I.** 1.92 M
- **J.** 0.64 M

**Answer: A**

*Source: mmlu_pro_chemistry | Category: science_knowledge | ID: science_knowledge_005*

---

## Q86. A 4 kHz audio signal is transmitted using PCM technique. If the system operation is to be just above the thresh-old and the output signal-to-noise ratio is equal to 47 dB, find N, the number of binary digits needed to assign individual binary code designation to the M quantization level. i.e., M = 2^N. Given that S_O = output signal power = (I^2 / T_s^2) \bullet (M^2S^2 / 12) N_q= quantization noise power = (I^2/T_s) \bullet (S^2/12)(2f_m) N_th = thermal noise power = (I^2/T_s^2) [(P_e2^2NS^2) / (3)] whereP_e= error probabilityand (2^2N+2P_e) = 0.26(1) (Note: Signal is sampled at 2f_s wheref_sis thenyquistrate).

- **A.** 12
- **B.** 10
- **C.** 6
- **D.** 7
- **E.** 5
- **F.** 11
- **G.** 4
- **H.** 9
- **I.** 13
- **J.** 8

**Answer: J**

*Source: mmlu_pro_engineering | Category: science_knowledge | ID: science_knowledge_006*

---

## Q87. Assume all gases are perfect unless stated otherwise. Note that 1 atm = 1.013 25 bar. Unless otherwise stated, thermochemical data are for 298.15 K.
Silylene $\left(\mathrm{SiH}_2\right)$ is a key intermediate in the thermal decomposition of silicon hydrides such as silane $\left(\mathrm{SiH}_4\right)$ and disilane $\left(\mathrm{Si}_2 \mathrm{H}_6\right)$. Moffat et al. (J. Phys. Chem. 95, 145 (1991)) report $\Delta_{\mathrm{f}} H^{\ominus}\left(\mathrm{SiH}_2\right)=+274 \mathrm{~kJ} \mathrm{~mol}^{-1}$. If $\Delta_{\mathrm{f}} H^{\ominus}\left(\mathrm{SiH}_4\right)=+34.3 \mathrm{~kJ} \mathrm{~mol}^{-1}$ and $\Delta_{\mathrm{f}} H^{\ominus}\left(\mathrm{Si}_2 \mathrm{H}_6\right)=+80.3 \mathrm{~kJ} \mathrm{~mol}^{-1}$(CRC Handbook (2008)), compute the standard enthalpies of the following reaction:
$\mathrm{Si}_2 \mathrm{H}_6(\mathrm{g}) \rightarrow \mathrm{SiH}_2(\mathrm{g})+\mathrm{SiH}_4(\mathrm{g})$

- **A.** 275$\mathrm{kJ} \mathrm{mol}^{-1}$
- **B.** 265$\mathrm{kJ} \mathrm{mol}^{-1}$
- **C.** 185$\mathrm{kJ} \mathrm{mol}^{-1}$
- **D.** 228$\mathrm{kJ} \mathrm{mol}^{-1}$
- **E.** 160$\mathrm{kJ} \mathrm{mol}^{-1}$
- **F.** 150$\mathrm{kJ} \mathrm{mol}^{-1}$
- **G.** 235$\mathrm{kJ} \mathrm{mol}^{-1}$
- **H.** 310$\mathrm{kJ} \mathrm{mol}^{-1}$
- **I.** 170$\mathrm{kJ} \mathrm{mol}^{-1}$
- **J.** 200$\mathrm{kJ} \mathrm{mol}^{-1}$

**Answer: D**

*Source: mmlu_pro_chemistry | Category: science_knowledge | ID: science_knowledge_007*

---

## Q88. The van der Waal equation is a modification of the ideal gas equation. It reads [P + (an^2 / V^2)] (V - nb) = nRT where P = pressure, V = volume, n = number of moles, R = gas constant, T = absolute temperature, and (a) and (b) are constants for a particular gas. The term an^2 /V^2 corrects the pressure for intermolecular attraction and the term - nb corrects the volume for molecular volume. Using this equation, determine whether a gas becomes more or less ideal when: (a.) the gas is compressed at constant temperature; (b.) more gas is added at constant volume and temperature; and (c.) The temperature of the gas is raised at constant volume.

- **A.** more ideal, less ideal, more ideal
- **B.** less ideal, more ideal, less ideal
- **C.** more ideal, less ideal, less ideal
- **D.** closer to being ideal, more ideal, less ideal
- **E.** less ideal, more ideal, more ideal
- **F.** less ideal, less ideal, less ideal
- **G.** more ideal, more ideal, less ideal
- **H.** less ideal, less ideal, closer to being ideal
- **I.** more ideal, less ideal, closer to being ideal
- **J.** closer to being ideal, less ideal, more ideal

**Answer: H**

*Source: mmlu_pro_chemistry | Category: science_knowledge | ID: science_knowledge_008*

---

## Q89. One of the methods for the synthesis of dilute acetic acid from ethanol is the use ofacetobacter. A dilute solution of ethanol is allowed to trickle down overbeechwoodshavings that have been inoculated with a culture of the bacteria. Air is forced through the vat countercurrent to the alcohol flow. The reaction can be written as C_2H_5OH (aq) + O_2 CH_3COOH (aq) + H_2O (l) acetic acid Calculate the Gibbs Free Energy, ∆G°, for this reaction, given the following Gibbs Free Energies of Formation: Ethanol = - 43.39 Kcal/mole, H_2O (l) = - 56.69 Kcal/mole, and Acetic acid = - 95.38 Kcal/mole.

- **A.** - 89.1 Kcal/mole
- **B.** - 65.5 Kcal/mole
- **C.** - 120.5 Kcal/mole
- **D.** - 20.3 Kcal/mole
- **E.** - 150.9 Kcal/mole
- **F.** - 56.69 Kcal/mole
- **G.** - 108.7 Kcal/mole
- **H.** - 95.38 Kcal/mole
- **I.** - 78.2 Kcal/mole
- **J.** - 43.39 Kcal/mole

**Answer: G**

*Source: mmlu_pro_chemistry | Category: science_knowledge | ID: science_knowledge_009*

---

## Q90. Nelson, et al. (Science 238, 1670 (1987)) examined several weakly bound gas-phase complexes of ammonia in search of examples in which the $\mathrm{H}$ atoms in $\mathrm{NH}_3$ formed hydrogen bonds, but found none. For example, they found that the complex of $\mathrm{NH}_3$ and $\mathrm{CO}_2$ has the carbon atom nearest the nitrogen (299 pm away): the $\mathrm{CO}_2$ molecule is at right angles to the $\mathrm{C}-\mathrm{N}$ 'bond', and the $\mathrm{H}$ atoms of $\mathrm{NH}_3$ are pointing away from the $\mathrm{CO}_2$. The magnitude of the permanent dipole moment of this complex is reported as $1.77 \mathrm{D}$. If the $\mathrm{N}$ and $\mathrm{C}$ atoms are the centres of the negative and positive charge distributions, respectively, what is the magnitude of those partial charges (as multiples of $e$ )?

- **A.** 0.150
- **B.** 0.123
- **C.** 0.135
- **D.** 0.098
- **E.** 0.157
- **F.** 0.091
- **G.** 0.112
- **H.** 0.176
- **I.** 0.204
- **J.** 0.085

**Answer: B**

*Source: mmlu_pro_chemistry | Category: science_knowledge | ID: science_knowledge_010*

---

## Q91. The discovery of the element argon by Lord Rayleigh and Sir William Ramsay had its origins in Rayleigh's measurements of the density of nitrogen with an eye toward accurate determination of its molar mass. Rayleigh prepared some samples of nitrogen by chemical reaction of nitrogencontaining compounds; under his standard conditions, a glass globe filled with this 'chemical nitrogen' had a mass of $2.2990 \mathrm{~g}$. He prepared other samples by removing oxygen, carbon dioxide, and water vapour from atmospheric air; under the same conditions, this 'atmospheric nitrogen' had a mass of $2.3102 \mathrm{~g}$ (Lord Rayleigh, Royal Institution Proceedings 14, 524 (1895)). With the hindsight of knowing accurate values for the molar masses of nitrogen and argon, compute the mole fraction of argon in the latter sample on the assumption that the former was pure nitrogen and the latter a mixture of nitrogen and argon.

- **A.** 0.022
- **B.** 0.011
- **C.** 0.003
- **D.** 0.030
- **E.** 0.008
- **F.** 0.025
- **G.** 0.040
- **H.** 0.015
- **I.** 0.018
- **J.** 0.005

**Answer: B**

*Source: mmlu_pro_chemistry | Category: science_knowledge | ID: science_knowledge_011*

---

## Q92. A 42-year-old man comes to the physician because of malaise, muscle and joint pain, and temperatures to 38.4°C (101.1°F) for 3 days. Three months ago, he underwent cadaveric renal transplantation resulting in immediate kidney function. At the time of discharge, his serum creatinine concentration was 0.8 mg/dL. He is receiving cyclosporine and corticosteroids. Examination shows no abnormalities. His leukocyte count is 2700/mm3 , and serum creatinine concentration is 1.6 mg/dL; serum cyclosporine concentration is in the therapeutic range. A biopsy of the transplanted kidney shows intracellular inclusion bodies. Which of the following is the most appropriate next step in management? 

- **A.** Increase the dosage of corticosteroids
- **B.** Begin ganciclovir therapy
- **C.** Decrease the dosage of cyclosporine
- **D.** Increase the dosage of cyclosporine
- **E.** Begin acyclovir therapy
- **F.** Begin amphotericin therapy
- **G.** Discontinue all medications and monitor the patient's condition
- **H.** Perform a second kidney transplant
- **I.** Decrease the dosage of corticosteroids
- **J.** Begin interferon therapy

**Answer: B**

*Source: mmlu_pro_health | Category: science_knowledge | ID: science_knowledge_012*

---

## Q93.  An automobile with a mass of $1000 \mathrm{~kg}$, including passengers, settles $1.0 \mathrm{~cm}$ closer to the road for every additional $100 \mathrm{~kg}$ of passengers. It is driven with a constant horizontal component of speed $20 \mathrm{~km} / \mathrm{h}$ over a washboard road with sinusoidal bumps. The amplitude and wavelength of the sine curve are $5.0 \mathrm{~cm}$ and $20 \mathrm{~cm}$, respectively. The distance between the front and back wheels is $2.4 \mathrm{~m}$. Find the amplitude of oscillation of the automobile, assuming it moves vertically as an undamped driven harmonic oscillator. Neglect the mass of the wheels and springs and assume that the wheels are always in contact with the road.


- **A.** -0.1 $ \mathrm{~mm}$
- **B.** -0.3 $\mathrm{~mm}$
- **C.** 0.05 $ \mathrm{~mm}$
- **D.** 0.25 $\mathrm{~mm}$
- **E.** -0.16 $ \mathrm{~mm}$
- **F.** -0.25 $\mathrm{~mm}$
- **G.** 0.2 $ \mathrm{~mm}$
- **H.** 1.0 $\mathrm{~mm}$
- **I.** 0.1 $\mathrm{~mm}$
- **J.** -0.5 $\mathrm{~mm}$

**Answer: E**

*Source: mmlu_pro_physics | Category: science_knowledge | ID: science_knowledge_013*

---

## Q94. A steel cylinder contains liquid at a mean bulk temperature of 80°F. Steam condensing at 212°F on the outside surface is used for heating the liquid. The coefficient of heat transfer on the steam side is 1,000 Btu/hr-ft^2-°F. The liquid is agitated by the stirring action of a turbine impeller. Its diameter is 2 ft., and it moves at an angular velocity of 100 rpm. The cylinder is 6 ft. long, with a diameter of 6 ft. and a wall thickness of 1/8 in. The thermal conductivity of steel may be taken as 9.4 Btu/hr-ft^2-°F. Properties of the liquid, taken as constant, are: c_p = 0.6 Btu/lbm-°Fk = 0.1 Btu/hr-ft-°F \rho = 60lbm/ft^3 The viscosity at 130°F is 653.4lbm/ft-hr, and at 212°F is 113.74lbm/ft-hr. Calculate the time required to raise the mean bulk temperature of the liquid to 180°F.

- **A.** 3.5 hr
- **B.** 1.92 hr
- **C.** 4.0 hr
- **D.** 1.75 hr
- **E.** 2.0 hr
- **F.** 3.0 hr
- **G.** 2.5 hr
- **H.** 1.5 hr
- **I.** 2.75 hr
- **J.** 2.15 hr

**Answer: B**

*Source: mmlu_pro_engineering | Category: science_knowledge | ID: science_knowledge_014*

---

## Q95. A silicious rock contains the mineral ZnS. To analyze for Zn, a sample of the rock is pulverized and treated with HCl to dissolve the ZnS (silicious matter is insoluable). Zinc is precipitated from solution by the addition of potassium ferrocyanide K_4 Fe (CN)_6. After filtering, the precipitate is dried and weighed. The reactions which occur are ZnS + 2HCl \rightarrow ZnCl_2 + H_2 S 2ZnCl_2 + K_4 Fe (CN)_6 \rightarrow Zn_2 Fe (CN)_6 + 4 KCl If a 2 gram sample of rock yields 0.969 gram of Zn_2Fe(CN)_6, what is the percentage of Zn in the sample? Atomic weight Zn = 65.4, molecular weight Zn_2 Fe (CN)_6 = 342.6.

- **A.** 10 % Zn
- **B.** 22 % Zn
- **C.** 20 % Zn
- **D.** 19 % Zn
- **E.** 15 % Zn
- **F.** 18 % Zn
- **G.** 16 % Zn
- **H.** 14 % Zn
- **I.** 12 % Zn
- **J.** 25 % Zn

**Answer: F**

*Source: mmlu_pro_chemistry | Category: science_knowledge | ID: science_knowledge_015*

---

## Q96. The combustion equation for octane burning in theoretical air (21% O_2 and 79% N_2) is C_8H_18(1) + 12.5O_2 + 12.5(3.76)N_2 \rightarrow 8CO_2 + 9H_2O + 47N_2 Determine the adiabatic flame temperature for liquid octane burningwith 200 percent theoretical air at 25°C. Use the followingdata to solve the problem: h^0_f = Standard enthalpy of formation (allenthalpies are in kcal/mol) DATA SPECIES h°_f (Kcal/mol) h25°C H139°C H117 / °C h838°C C_8H_18 - 27093.8 --- --- --- --- C0_2 - 42661.1 1015.6 8771 7355 5297 H_2O - 26218.1 1075.5 7153 6051 4395 N_2 0 940.0 5736 4893 3663 O_2 0 938.7 6002 5118 3821

- **A.** 1,100°C
- **B.** 1,450°C
- **C.** 1,171°C
- **D.** 1,610°C
- **E.** 1,327°C
- **F.** 1,500°C
- **G.** 1,233°C
- **H.** 1,050°C
- **I.** 1,300°C
- **J.** 1,393°C

**Answer: G**

*Source: mmlu_pro_engineering | Category: science_knowledge | ID: science_knowledge_016*

---

## Q97. Two rods of the same diameter, one made of brass and of length 25 cm, the other of steel and of length50 cm, are 50 cm, are placed end to end and pinned to two rigid supports. The temperature of the rods rises to 40°C. What is the stress in each rod? Young'smodulifor steel and brass are 20 × 10^11 steel and brass are 20 × 10^11 dynes \textbullet cm^\rule{1em}{1pt}2 and 10 × 10^11 dynes \textbullet cm ^-2, respectively, and 10^11 dynes \textbullet cm ^-2, respectively, and their respective coefficients of expansion are 1.2 × 10 ^-5 C coefficients of expansion are 1.2 × 10 ^-5 C deg^-1 and 1.8 × 10 ^-5 C deg ^-1. 1.8 × 10 ^-5 C deg ^-1.

- **A.** 0.50 × 10^9 dyne . cm^-2
- **B.** 1.2 × 10^9 dyne . cm^-2
- **C.** 1.0 × 10^9 dyne . cm^-2
- **D.** 2.5 × 10^9 dyne . cm^-2
- **E.** 1.84 × 10^9 dyne . cm^-2
- **F.** 1.5 × 10^9 dyne . cm^-2
- **G.** 2.0 × 10^9 dyne . cm^-2
- **H.** 0.60 × 10^9 dyne . cm^-2
- **I.** 0.30 × 10^9 dyne . cm^-2
- **J.** 0.84 × 10^9 dyne . cm^-2

**Answer: J**

*Source: mmlu_pro_physics | Category: science_knowledge | ID: science_knowledge_017*

---

## Q98. A cylindrical container 3 ft. in diameter and 5 ft. high has oil. A transformer is immersed in the oil. Determine the surface temperatureof the container if the loss of energy is 2.0 kW. Assume the bottom of the container to be insulated and that theloss is only by natural convection to the ambient air at 80°F. TABLE Geometry Range of application C n L Vertical planes and cylinders 10^4 < N(Gr)LNPr< 10^9 N 10^9 < N(Gr)LNPr< 10^12 N 0.29 0.19 1/4 1/3 height 1 Horizontal cylinders 10^3 < N(Gr)LNPr< 10^9 N 10^9 < N(Gr)LNPr< 10^12 N 10^5 < N(Gr)LNPr< 2×10^7 N 0.27 0.18 0.27 1/4 1/3 1/4 diameter 1 length of side Horizontal plates - heated plates facing up or cooled plates facing down Cooled plates facing up or heated plates facing down 2×10^7 < N(Gr)LNPr< 3×10^10 N 3×10^5 < N(Gr)LNPr< 3×10^10 N 0.22 0.12 1/3 1/4 1 length of side

- **A.** 280°F
- **B.** 310°F
- **C.** 300°F
- **D.** 265°F
- **E.** 275°F
- **F.** 285°F
- **G.** 260°F
- **H.** 273°F
- **I.** 290°F
- **J.** 250°F

**Answer: H**

*Source: mmlu_pro_engineering | Category: science_knowledge | ID: science_knowledge_018*

---

## Q99. Assume that all gases are perfect and that data refer to 298.15 K unless otherwise stated. Consider a system consisting of $2.0 \mathrm{~mol} \mathrm{CO}_2(\mathrm{~g})$, initially at $25^{\circ} \mathrm{C}$ and $10 \mathrm{~atm}$ and confined to a cylinder of cross-section $10.0 \mathrm{~cm}^2$. It is allowed to expand adiabatically against an external pressure of 1.0 atm until the piston has moved outwards through $20 \mathrm{~cm}$. Assume that carbon dioxide may be considered a perfect gas with $C_{V, \mathrm{~m}}=28.8 \mathrm{~J} \mathrm{~K}^{-1} \mathrm{~mol}^{-1}$ and calculate $\Delta T$.

- **A.** -0.645 K
- **B.** -0.562$\text{K}$
- **C.** -0.410 K
- **D.** -0.223$\text{K}$
- **E.** -0.500 K
- **F.** -0.295 K
- **G.** -0.347$\text{K}$ 
- **H.** -0.150 K
- **I.** -0.725 K
- **J.** -0.479$\text{K}$

**Answer: G**

*Source: mmlu_pro_chemistry | Category: science_knowledge | ID: science_knowledge_019*

---

## Q100. A physician is conducting a retrospective review of a trial involving the use of Drug X in patients with a specific disease. It is known that Drug X is associated with an increased probability of cancer in patients who use the drug. A total of 600 individuals with a specific disease were included in the trial. Of the participants, 200 individuals received Drug X and 400 individuals did not receive it. One hundred individuals who received Drug X died of a particular type of cancer and 100 individuals who did not receive the drug died of the same type of cancer. Based on these data, which of the following is the relative risk of death from this type of cancer in individuals who take Drug X as compared with individuals who do not take Drug X? 

- **A.** Individuals who take Drug X have zero risk of dying from this type of cancer
- **B.** Individuals who take Drug X have four times the risk of dying from this type of cancer
- **C.** Individuals who take Drug X have half the risk of dying from this type of cancer
- **D.** Individuals who take Drug X have an equal risk of dying from this type of cancer
- **E.** Individuals who take Drug X have six times the risk of dying from this type of cancer
- **F.** Individuals who take Drug X have three times the risk of dying from this type of cancer
- **G.** Individuals who do not take Drug X have three times the risk of dying from this type of cancer
- **H.** Individuals who do not take Drug X have two times the risk of dying from this type of cancer
- **I.** Individuals who take Drug X have two times the risk of dying from this type of cancer
- **J.** Individuals who take Drug X have five times the risk of dying from this type of cancer

**Answer: I**

*Source: mmlu_pro_health | Category: science_knowledge | ID: science_knowledge_020*

---

## Q101. A 25-year-old man comes to the emergency department because he developed chest pain and shortness of breath 1 hour ago, shortly after snorting cocaine for the first time. He rates the chest pain as a 7 on a 10-point scale and notes that the pain is radiating down his left arm. Medical history is unremarkable and the patient takes no medications or any other illicit drugs. He is 178 cm (5 ft 10 in) tall and weighs 70 kg (154 lb); BMI is 22 kg/m2 . The patient is diaphoretic. Vital signs are temperature 37.5°C (99.5°F), pulse 110/min, respirations 16/min, and blood pressure 200/100 mm Hg. Pulse oximetry on room air shows an oxygen saturation of 95%. Pupils are equal, round, and reactive to light and accommodation. Lungs are clear to auscultation and percussion. Auscultation of the heart discloses an audible S1 and S2. There is no edema, cyanosis, or clubbing of the digits. The patient is fully oriented. He is treated with supplemental oxygen, a 325-mg aspirin tablet, and intravenous nitroglycerin and lorazepam. Despite therapy, he continues to have chest pain and shortness of breath. ECG shows sinus tachycardia with no ST-segment or T-wave abnormalities. Which of the following is the most appropriate additional pharmacotherapy to initiate at this time?

- **A.** Nitroprusside
- **B.** Verapamil
- **C.** Alteplase
- **D.** Furosemide
- **E.** Phentolamine
- **F.** Atorvastatin
- **G.** Carvedilol
- **H.** Metoprolol
- **I.** Lisinopril
- **J.** Warfarin

**Answer: E**

*Source: mmlu_pro_health | Category: science_knowledge | ID: science_knowledge_021*

---

## Q102. While you are on rounds at a local nursing facility, the nurse mentions that your patient, a 79-year-old woman, appears to be a "poor eater." She was admitted to the nursing facility 3 months ago from the hospital where she was treated for congestive heart failure. Her daughter had moved away from the area, and nursing home placement was necessary because the patient could no longer function independently. Her present medications include furosemide and digoxin. Physical examination is normal except for a weight loss of 3.5 kg (7 lb) during the past 3 months. In your conversation with the patient, she says, "No, I'm not depressed, I just don't have an appetite anymore. Nothing tastes good to me. I have a little bit of nausea most of the time." Which of the following is the most appropriate initial diagnostic study?

- **A.** Abdominal ultrasound
- **B.** Determination of serum albumin concentration
- **C.** Determination of serum sodium level
- **D.** Electrocardiogram
- **E.** Determination of serum digoxin level
- **F.** Renal function tests
- **G.** Complete blood count
- **H.** Chest x-ray
- **I.** Thyroid function tests
- **J.** Stool sample for occult blood

**Answer: E**

*Source: mmlu_pro_health | Category: science_knowledge | ID: science_knowledge_022*

---

## Q103. A 67-year-old man with Parkinson disease is admitted to the hospital for treatment of pneumonia. The patient's daughter, who is visiting the patient, says he has had increased lethargy for the past day and decreased ambulation during the past 6 months. She also says that there are times during the day when his tremors increase in severity, although he continues to care for himself at home. Medical history is also remarkable for hypertension. Medications include hydrochlorothiazide, atenolol, levodopa, and carbidopa. He is 168 cm (5 ft 6 in) tall and weighs 78 kg (172 lb); BMI is 28 kg/m2 . Vital signs are temperature 38.9°C (102.0°F), pulse 60/min supine and 68/min standing, respirations 22/min, and blood pressure 100/60 mm Hg supine and 80/50 mm Hg standing. The patient appears ill and older than his stated age. He is fully oriented but lethargic. Auscultation of the chest discloses rhonchi in the right mid lung field. Abdominal examination discloses no abnormalities. Neurologic examination discloses masked facies, bradykinesia, and cogwheel rigidity; gait was not assessed on admission. Chest x-ray shows a right lower lobe infiltrate. ECG shows no abnormalities. Appropriate intravenous antibiotic therapy is initiated. Prior to discharge, which of the following is the most appropriate step? 

- **A.** Place a percutaneous endoscopic gastrostomy (PEG) tube
- **B.** Obtain CT scan of the chest
- **C.** Begin corticosteroid treatment
- **D.** Start patient on anticoagulant therapy
- **E.** Administer influenza vaccine
- **F.** Arrange for home oxygen therapy
- **G.** Prescribe fludrocortisone
- **H.** Discontinue levodopa and carbidopa
- **I.** Initiate physical therapy
- **J.** Obtain a swallowing evaluation

**Answer: J**

*Source: mmlu_pro_health | Category: science_knowledge | ID: science_knowledge_023*

---

## Q104. A lunar module usedAerozine50 as fuel and nitrogen tetroxide (N_2 O_4, molecular weight = 92.0 g/mole) as oxidizer.Aerozine50 consists of 50 % by weight of hydrazine (N_2 H_4, molecular weight = 32.0 g/mole) and 50 % by weight of unsymmetricaldimethylhydrazine ((CH_3)_2 N_2 H_2, molecular weight = 60.0 g/mole). The chief exhaust product was water (H_2 O, molecular weight = 18.0 g/mole). Two of the reactions that led to the formation of water are the following: 2N_2 H_4 + N_2 O_4\rightarrow3N_2 + 4H_2 O (CH_3)_2 N_2 H_2 + 2N_2 O_4\rightarrow2CO_2 + 3N_2 + 4H_2 O. If we assume that these reactions were the only ones in which water was formed, how much water was produced by the ascent of the lunar module if 2200 kg ofAerozine50 were consumed in the process?

- **A.** 1.5 × 10^3 kg of water
- **B.** 3.0 × 10^3 kg of water
- **C.** 2.5 × 10^3 kg of water
- **D.** 4.0 × 10^3 kg of water
- **E.** 1.8 × 10^3 kg of water
- **F.** 2.2 × 10^3 kg of water
- **G.** 3.5 × 10^3 kg of water
- **H.** 2.0 × 10^3 kg of water
- **I.** 1.0 × 10^3 kg of water
- **J.** 2.8 × 10^3 kg of water

**Answer: C**

*Source: mmlu_pro_chemistry | Category: science_knowledge | ID: science_knowledge_024*

---

## Q105. A previously healthy 15-year-old boy is brought to the emergency department in August 1 hour after the onset of headache, dizziness, nausea, and one episode of vomiting. His symptoms began during the first hour of full-contact football practice in full uniform. He reported feeling weak and faint but did not lose consciousness. He vomited once after drinking water. On arrival, he is diaphoretic. He is not oriented to person, place, or time. His temperature is 39.5°C (103.1°F), pulse is 120/min, respirations are 40/min, and blood pressure is 90/65 mm Hg. Examination, including neurologic examination, shows no other abnormalities. Which of the following is the most appropriate next step in management? 

- **A.** Administer sodium chloride tablets
- **B.** Immerse the patient in an ice water bath
- **C.** Administer a dose of ibuprofen
- **D.** Administer a glucose injection
- **E.** Apply cold compresses to the forehead
- **F.** Administer intravenous fluids
- **G.** Administer an epinephrine injection
- **H.** Perform a lumbar puncture
- **I.** Administer oxygen via a nasal cannula
- **J.** Obtain a CT scan of the head

**Answer: B**

*Source: mmlu_pro_health | Category: science_knowledge | ID: science_knowledge_025*

---

## Q106. A 25-year-old woman comes to the physician because of a 2-month history of numbness in her right hand. During this period, she has had tingling in the right ring and small fingers most of the time. She has no history of serious illness and takes no medications. She is employed as a cashier and uses a computer at home. She played as a pitcher in a softball league for 5 years until she stopped 2 years ago. Vital signs are within normal limits. Examination shows full muscle strength. Palpation of the right elbow produces a jolt of severe pain in the right ring and small fingers. Sensation to pinprick and light touch is decreased over the medial half of the right ring finger and the entire small finger. The most likely cause of these findings is entrapment of which of the following on the right?

- **A.** Median nerve at the forearm
- **B.** Radial nerve at the forearm
- **C.** Ulnar nerve at the elbow
- **D.** Median nerve at the elbow
- **E.** Radial nerve at the wrist
- **F.** Musculocutaneous nerve at the forearm
- **G.** Radial nerve at the elbow
- **H.** Ulnar nerve at the wrist
- **I.** Median nerve at the wrist
- **J.** Musculocutaneous nerve at the wrist

**Answer: C**

*Source: mmlu_pro_health | Category: science_knowledge | ID: science_knowledge_026*

---

## Q107. A 44-year-old woman with a 10-year history of arthritis comes to the office because she has had increasing pain and stiffness in her hands, wrists, and knees during the past several months. She also has had increasing fatigue for the past month, along with a weight loss of 1.8 to 2.2 kg (4 to 5 lb). She has seen numerous physicians for her arthritis in the past and has tried various medications and devices, including copper bracelets from Mexico given to her by friends. Review of her medical records confirms that the initial diagnosis of rheumatoid arthritis is correct. She says, "I had several drop attacks during the past 3 months." She characterizes these attacks as episodes of weakness and loss of feeling in her legs for several minutes. During one of these episodes, she became incontinent. She currently takes aspirin approximately four times daily and ibuprofen occasionally. Physical examination shows facial plethora and swollen and painful metacarpophalangeal and knee joints, bilaterally. There is moderate ulnar deviation of the fingers. The remainder of the examination discloses no abnormalities. Which of the following is the most likely cause of her "drop attacks?"

- **A.** Side effect of the copper bracelets
- **B.** Spinal stenosis
- **C.** Reaction to aspirin or ibuprofen
- **D.** Cardiac arrhythmia
- **E.** Anxiety
- **F.** Neurological side effects of rheumatoid arthritis
- **G.** Transient ischemic attack
- **H.** Adrenal insufficiency
- **I.** Hypoglycemia
- **J.** Atlanto-axial instability

**Answer: J**

*Source: mmlu_pro_health | Category: science_knowledge | ID: science_knowledge_027*

---

## Q108. A 72-year-old man comes to the physician because of a 7-month history of leg weakness and dry eyes and mouth. He also has had a 10.4-kg (23-lb) weight loss over the past 4 months despite no change in appetite. He has smoked one and a half packs of cigarettes daily for 50 years. He drinks 4 oz of alcohol daily. He has peptic ulcer disease and emphysema. Medications include cimetidine, theophylline, and low-dose prednisone. Examination shows mild ptosis. He has a barrelshaped chest. Breath sounds are distant. There is moderate weakness of proximal muscles of the lower extremities. Reflexes are absent. He has difficulty rising from a chair. Sensory examination shows no abnormalities. An x-ray shows a hyperinflated chest and a 3 x 4-cm mass in the right hilum. His neurologic findings are most likely due to a lesion involving which of the following?

- **A.** Presynaptic neuromuscular junction
- **B.** Spinal cord
- **C.** Muscle membrane
- **D.** Central nervous system
- **E.** Parasympathetic nervous system
- **F.** Motor cortex
- **G.** Postsynaptic neuromuscular junction
- **H.** Peripheral nerve
- **I.** Sympathetic nervous system
- **J.** Sensory nerve

**Answer: A**

*Source: mmlu_pro_health | Category: science_knowledge | ID: science_knowledge_028*

---

## Q109. A 14-year-old girl is brought to the physician after her mother learned that she began having sexual intercourse with various partners 1 month ago. She does not use condoms or other contraception. The mother is concerned about her behavior. The patient's parents separated 3 months ago. She had been an honor student and excelled in sports and leadership positions at school before the separation. Since the separation, however, she has become sullen, defiant, and rebellious. She has begun smoking cigarettes, disobeying her curfew, and being truant from school. This patient is most likely using which of the following defense mechanisms?

- **A.** Acting out
- **B.** Intellectualization
- **C.** Projection
- **D.** Regression
- **E.** Displacement
- **F.** Rationalization
- **G.** Denial
- **H.** Repression
- **I.** Sublimation
- **J.** Reaction formation

**Answer: A**

*Source: mmlu_pro_health | Category: science_knowledge | ID: science_knowledge_029*

---

## Q110. Find the sublimation rate of a uranium hexafluoride UF_6 cylinder7 mm. diameter when exposed to an air stream flowingat a velocity of 3.5 m/s. The bulk air is at 65°C and 1 atm. pressure.The surface temperature of the solidis 40°C at which its vapor pressure is 410 mm. Hg (54.65kN/m^2). The average heat transfer coefficient of fluid flowing perpendicularto a circular cylinder for fluid Reynolds number between1 and 4000 is given by Nu = 0.43 + 0.532 (Re)^0.5(Pr)^0.31 whereNu and Re are calculated with the cylinder diameter andfluid properties at mean temperature of cylinder and bulk-fluid.

- **A.** 29.78 kmolUF_6/m^2.s
- **B.** 0.731 kmolUF_6/m^2.s
- **C.** 1.442 × 10-3kmol/m2.sec. kmol/m
- **D.** 5.23 × 10^-4 kmolUF_6/m^2.s
- **E.** 0.850 × 10^-3 kmolUF_6/m^2.s
- **F.** 2.56 × 10^-3 kmolUF_6/m^2.s
- **G.** 0.415 × 10^-2 kmolUF_6/m^2.s
- **H.** 3.67 × 10^-3 kmolUF_6/m^2.s
- **I.** 1.789 × 10^-3 kmolUF_6/m^2.s
- **J.** 1.12 × 10^-3kmolUF_6/m^2.s

**Answer: J**

*Source: mmlu_pro_engineering | Category: science_knowledge | ID: science_knowledge_030*

---

## Q111. For a steady, turbulent, constant property, two dimensional boundarylayer-type flow over a flat plate at zero angle of approach, the velocity profile is given by u = v(y/\delta)^1^/7(1) where v = free stream velocity \delta = boundary layer thickness andthe local skin friction coefficient is given by C_f= [\tau / {(1/2)\rhoV^2}] = 0.045(ѵ /\rhoV)^1/4(2) where \tau = local shear stress \rho = density of the flowing fluid ѵ = kinematic viscosity Determine the local boundary layer thickness \delta, as a functionof x, by substituting the given equations into the integralform of the momentum equation.

- **A.** \(\delta= \frac{(0.37x)^2}{(Re_x)^{1/5}}\)
- **B.** \(\delta= \frac{(0.37x)}{(Re_x)^{1/3}}\)
- **C.** \delta= {(0.37x) / (Re_x)^1^/6}
- **D.** \(\delta= \frac{(0.47x)}{(Re_x)^{1/5}}\)
- **E.** \(\delta= \frac{(0.37x)}{Re_x}\)
- **F.** \(\delta= \frac{(0.50x)}{(Re_x)^{1/5}}\)
- **G.** \delta= {(0.37x) / (Re_x)^1^/5}
- **H.** \delta= {(0.37x) / (Re_x)^1^/4}
- **I.** \(\delta= \frac{(0.30x)}{(Re_x)^{1/7}}\)
- **J.** \delta= {(0.72x) / (Re_x)^1^/5}

**Answer: G**

*Source: mmlu_pro_engineering | Category: science_knowledge | ID: science_knowledge_031*

---

## Q112. Compute the regulation and efficiency at full load, 80 power factor, lagging current, of the 15-kva, 2,400: 240-volt, 60 - distribution transformer to which the following data apply. (Subscript H means high-voltage, subscript X means low-voltage winding. Short-circuit test Open-circuit test V_H = 74.5 v V_X = 240 v I_H = 6.25 amp I_X = 1.70 amp P_H = 237 watts P_X = 84 watts Frequency = 60 \texttheta = 25 C Frequency = 60 \sim Direct-current resistances measured at 25° C R_dcH = 2.80 ohmsR_dcX = 0,0276 ohm The data given above have been corrected for instrument losses where this correction was necessary.

- **A.** 0.9801, 5.09%
- **B.** 0.9604, 7.12%
- **C.** 0.9505, 4.07%
- **D.** 0.8899, 5.55%
- **E.** 0.9007, 2.05%
- **F.** 0.9406, 2.96%
- **G.** 0.9302, 1.98%
- **H.** 0.9203, 6.15%
- **I.** 0.9950, 4.50%
- **J.** 0.9709, 3.08%

**Answer: J**

*Source: mmlu_pro_engineering | Category: science_knowledge | ID: science_knowledge_032*

---

## Q113. In an orienteering class, you have the goal of moving as far (straight-line distance) from base camp as possible by making three straight-line moves. You may use the following displacements in any order: (a) $\vec{a}, 2.0 \mathrm{~km}$ due east (directly toward the east); (b) $\vec{b}, 2.0 \mathrm{~km} 30^{\circ}$ north of east (at an angle of $30^{\circ}$ toward the north from due east); (c) $\vec{c}, 1.0 \mathrm{~km}$ due west. Alternatively, you may substitute either $-\vec{b}$ for $\vec{b}$ or $-\vec{c}$ for $\vec{c}$. What is the greatest distance you can be from base camp at the end of the third displacement? (We are not concerned about the direction.)

- **A.** 4.2 m
- **B.** 4.6 km
- **C.** 3.8 km
- **D.** 2.9 km
- **E.** 5.2 km
- **F.** 3.5 m
- **G.** 4.8 m
- **H.** 5.0 km
- **I.** 5.6 m
- **J.** 3.2 km

**Answer: G**

*Source: mmlu_pro_physics | Category: science_knowledge | ID: science_knowledge_033*

---

## Q114. Assume all gases are perfect unless stated otherwise. Unless otherwise stated, thermodynamic data are for 298.15 K.  Given the reactions (1) and (2) below, determine $\Delta_{\mathrm{r}} U^{\ominus}$ for reaction (3).

(1) $\mathrm{H}_2(\mathrm{g})+\mathrm{Cl}_2(\mathrm{g}) \rightarrow 2 \mathrm{HCl}(\mathrm{g})$, $\Delta_{\mathrm{r}} H^{\ominus}=-184.62 \mathrm{~kJ} \mathrm{~mol}^{-1}$

(2) $2 \mathrm{H}_2(\mathrm{g})+\mathrm{O}_2(\mathrm{g}) \rightarrow 2 \mathrm{H}_2 \mathrm{O}(\mathrm{g})$, $\Delta_{\mathrm{r}} H^\ominus=-483.64 \mathrm{~kJ} \mathrm{~mol}^{-1}$

(3) $4 \mathrm{HCl}(\mathrm{g})+\mathrm{O}_2(\mathrm{g}) \rightarrow 2 \mathrm{Cl}_2(\mathrm{g})+2 \mathrm{H}_2 \mathrm{O}(\mathrm{g})$

- **A.** -150.72$\mathrm{kJ} \mathrm{mol}^{-1}$
- **B.** -200.36$\mathrm{kJ} \mathrm{mol}^{-1}$
- **C.** -135.50 kJ mol^{-1}
- **D.** -220.14 kJ mol^{-1}
- **E.** -162.80 kJ mol^{-1}
- **F.** -89.10 kJ mol^{-1}
- **G.** -97.66 kJ mol^{-1}
- **H.** -111.92$\mathrm{kJ} \mathrm{mol}^{-1}$
- **I.** -175.88 kJ mol^{-1}
- **J.** -125.48$\mathrm{kJ} \mathrm{mol}^{-1}$

**Answer: H**

*Source: mmlu_pro_chemistry | Category: science_knowledge | ID: science_knowledge_034*

---

## Q115. A 30-year-old nulliparous female presents to the office with the complaint of mood changes. She says that for the past several months she has been anxious, hyperactive, and unable to sleep 3 to 4 days prior to the onset of menses. She further reports that on the day her menses begins she becomes acutely depressed, anorectic, irritable, and lethargic. She has no psychiatric history. Physical examination findings are normal. She and her husband have been trying to conceive for over 2 years. History reveals a tuboplasty approximately 1 year ago to correct a closed fallopian tube. The most likely diagnosis is 

- **A.** Major depressive disorder
- **B.** Seasonal affective disorder
- **C.** cyclothymic personality
- **D.** generalized anxiety disorder
- **E.** adjustment disorder with depressed mood
- **F.** Bipolar II disorder
- **G.** bipolar I disorder, mixed
- **H.** Post-traumatic stress disorder
- **I.** Persistent depressive disorder (dysthymia)
- **J.** Premenstrual dysphoric disorder

**Answer: E**

*Source: mmlu_pro_health | Category: science_knowledge | ID: science_knowledge_035*

---

## Q116. The mean ionic activity coefficient of an electrolyte in a dilute solution is given approximately by ln \gamma_\pm = - A\surd(m) where m is the molality and A is a constant. Using the Gibbs-Duhem equation find the activity coefficient \gamma_1 of the solvent in this dilute solution in terms of A, m, and constants characteristic of the solvent and solute. Ob-serve that ln X_1 = ln [(n_1) / (n_1 + mv)] = ln [(1) / {(1 + mv) / n_1}] \approx - [(mv) / (n_1)] where v is the number of ions in the formula (for example v = 2 for NaCl) and n_1 is the number of moles of solvent in 1000 g of solvent. All quantities except \gamma_1 , the un-known, can thus be expressed in terms of the variable m.

- **A.** \gamma_1 = e[{(Av^2) / (3n_1)} m^3/2]
- **B.** \gamma_1 = e[{(Av^2) / (2n_1)} m^3/2]
- **C.** \gamma_1 = e[{(Av) / (3n_1)} m^3/2]
- **D.** \gamma_1 = e[{(Av) / (3n_1^2)} m^3/2]
- **E.** \gamma_1 = [{(Av) / (3n_1)} m^3/2]
- **F.** \gamma_1 = [{(Av^2) / (3n_1)} m^3/2]
- **G.** \gamma_1 = [{(A) / (n_1v)} m^3/2]
- **H.** \gamma_1 = e[{(A) / (2n_1v^2)} m^3/2]
- **I.** \gamma_1 = e[{(A) / (3n_1v)} m^3/2]
- **J.** \gamma_1 = [{(Av) / (2n_1)} m^3/2]

**Answer: C**

*Source: mmlu_pro_chemistry | Category: science_knowledge | ID: science_knowledge_036*

---

## Q117. For point-to-point communication at higher frequencies, the desiredradiation pattern is a single narrow lobe or beam. To obtainsuch a characteristic (at least approximately) a multi-elementlinear array is usually used. An array is linear whenthe elements of the ar-ray are spaced equally along a straightline. In a uniform linear array the elements are fed withcurrents of equal magnitude and having a uniform progressivephase shift along the line. The pattern of such anarray can be obtained by addingvectoriallythe field strengthsdue to each of the elements. For a uniform array of non-directionalelements the field strength would be E_T = E_0 \vert1 +e^J\psi+ e^J2\psi + e^J3\psi + ... +e^j^(^n-1)\psi \vert(1) where \psi =\betadcos\textphi + \alpha and\alpha is the progressive phase shift between elements, (\alpha is theangle by which the current in any element leads the currentin the preceding element.) Compute the pattern of such a linear array.

- **A.** 0.512
- **B.** 0.612
- **C.** 0.212
- **D.** 0.212 (with a different phase shift)
- **E.** 0.412
- **F.** 0.312
- **G.** 0.112
- **H.** 0.712
- **I.** 0.912
- **J.** 0.112 (with a different element spacing)

**Answer: C**

*Source: mmlu_pro_engineering | Category: science_knowledge | ID: science_knowledge_037*

---

## Q118. A 16-year-old girl is brought to the office by her mother because she is concerned that she may have contracted meningitis from her soccer teammate, who was diagnosed with meningococcal meningitis and admitted to the hospital yesterday. The patient's soccer team traveled to a neighboring state to participate in a tournament 1 week ago and she shared a hotel room with the girl who was hospitalized. The patient feels well but is concerned she may have "caught the same bug." Medical history is remarkable for asthma. Medications include inhaled albuterol. Vital signs are temperature 37.2°C (98.9°F), pulse 64/min, respirations 16/min, and blood pressure 107/58 mm Hg. Physical examination shows no abnormalities. Which of the following is the most appropriate intervention for this patient at this time?

- **A.** Administer a tetanus shot
- **B.** Prescribe azithromycin
- **C.** Administer the meningococcal vaccine
- **D.** Prescribe ibuprofen
- **E.** Prescribe penicillin
- **F.** Administer the pneumococcal vaccine
- **G.** Prescribe amoxicillin
- **H.** Prescribe doxycycline
- **I.** Prescribe rifampin
- **J.** No intervention is needed at this time

**Answer: I**

*Source: mmlu_pro_health | Category: science_knowledge | ID: science_knowledge_038*

---

## Q119. A 15-year-old girl is brought to the physician 3 months after she had a blood pressure of 150/95 mm Hg at a routine examination prior to participation in school sports. She is asymptomatic and has no history of serious illness. Twelve months ago, she was diagnosed with a urinary tract infection and treated with oral trimethoprim-sulfamethoxazole. She currently takes no medications. Subsequent blood pressure measurements on three separate occasions since the last visit have been: 155/94 mm Hg, 145/90 mm Hg, and 150/92 mm Hg. She is at the 50th percentile for height and 95th percentile for weight. Her blood pressure today is 150/90 mm Hg confirmed by a second measurement, pulse is 80/min, and respirations are 12/min. Examination shows no other abnormalities. Her hematocrit is 40%. Urinalysis is within normal limits. Cardiac and renal ultrasonography shows no abnormalities. Which of the following is the most appropriate next step in management?

- **A.** Measurement of serum potassium levels
- **B.** Measurement of urine corticosteroid concentrations
- **C.** Exercise and weight reduction program
- **D.** Captopril therapy
- **E.** Initiate diuretic therapy
- **F.** Referral for psychiatric evaluation
- **G.** Initiate calcium channel blocker therapy
- **H.** Measurement of urine catecholamine concentrations
- **I.** Measurement of plasma renin activity
- **J.** Initiate beta-blocker therapy

**Answer: C**

*Source: mmlu_pro_health | Category: science_knowledge | ID: science_knowledge_039*

---

## Q120. A 15-year-old girl comes to the emergency department because, she says, "something has been sticking out of my bottom since I had a bowel movement this morning." She has not had previous episodes, although for more than 1 year she has had occasional difficulty passing stools. She is not in pain but is afraid to move her bowels for fear that the problem will worsen. She tells you that she moved away from home more than a year ago and that her parents contribute nothing to her support. She has a 6-month-old child and lives with a 28-year-old female cousin. She has never been married and does not work or attend school. She has no other symptoms. In order to follow the correct procedure for treating a minor, which of the following is the most appropriate step prior to evaluating this patient's rectal problem?

- **A.** Obtain written consent from her 28-year-old cousin
- **B.** Obtain written consent from at least one of her parents
- **C.** Obtain a court order permitting evaluation
- **D.** Obtain the written consent of at least two licensed physicians
- **E.** Obtain the girl's consent in the presence of a witness
- **F.** Obtain written consent from the child's father
- **G.** Obtain verbal consent from at least one of her parents
- **H.** Wait until the girl turns 16 to perform the evaluation
- **I.** Accept the girl's consent as sufficient
- **J.** Obtain written consent from a social worker

**Answer: I**

*Source: mmlu_pro_health | Category: science_knowledge | ID: science_knowledge_040*

---

## Q121. A supplier of ink for printers sent the following letter to all of its customers:"Closeout special! We have decided to no longer stock green ink cartridges. We have on hand a limited supply of green ink cartridges for all printers; when they're gone, they're gone! Please submit your orders as soon as possible to make sure your order can be filled. "One of the regular customers of the supplier sent the following reply by fax:"Sorry to hear that you will no longer carry green ink cartridges, since that is one of our favorite colors. Please ship 100 green ink cartridges to our office as soon as possible. "The supplier faxed an acknowledgement of the order to the customer with a promise that the cartridges would be shipped out in one week. The next day, the supplier received the following e-mail from the customer:"Please cancel our order. We just discovered that we already have plenty of green ink cartridges in inventory. " The supplier proceeded to sell its entire stock of green ink cartridges at its asking price to other customers. In an action for breach of contract by the supplier against the customer, what is the maximum amount of damages that the supplier should be entitled to recover?

- **A.** Consequential damages, since the green ink cartridges were unique because they were the last of their kind to be offered for sale by the supplier.
- **B.** The cost of the ink cartridges plus any loss in profit from the potential sale to other customers.
- **C.** $10,000, which is double the asking price of the cartridges, as a penalty for the customer's late cancellation.
- **D.** Only incidental damages, if any, that the supplier has incurred in preparing the green ink cartridges for shipment to the customer before receiving the customer's e-mail.
- **E.** The cost of the ink cartridges plus the cost of shipping, as the supplier had already promised to ship them out.
- **F.** $5,000, which was the asking price for the 100 green ink cartridges ordered.
- **G.** The cost of the ink cartridges plus any loss in profit from the sale to other customers, since the supplier had to sell the cartridges at a lower price.
- **H.** The full cost of the cartridges plus any additional costs incurred in the sale to other customers.
- **I.** Nothing.
- **J.** Any additional costs incurred by the supplier in obtaining replacement cartridges to fulfill the customer's order.

**Answer: D**

*Source: mmlu_pro_law | Category: humanities | ID: humanities_001*

---

## Q122. A nephew inherited a large parcel of unimproved land from his uncle. In need of cash, the nephew decided to sell the parcel. He contacted a real estate agent in the area about listing the parcel for sale. The nephew and the agent entered into a valid written contract whereby the agent promised to undertake best efforts to find a buyer for the parcel. The contract also provided that the agent was to be paid a commission of 5 percent on the gross sale price following the consummation of the sale and transfer of title. The agent succeeded in finding a buyer for the parcel. The agent notified the nephew that he had found a developer who wanted to purchase the parcel for $500,000. The agent handed the nephew a real estate sales contract, signed by the developer, in which the developer agreed to pay $500,000 for the purchase of the parcel. The nephew then signed the agreement himself. However, before consummation of the sale and transfer of title, the developer, without cause, repudiated the contract. Despite the agent's insistence, the nephew refused to either sue the developer to enforce the land sale contract or pay the agent his commission. If the agent sues the nephew for breach of the brokerage agreement, which of the following, if anything, is the agent's proper measure of recovery?

- **A.** Nothing, because the nephew did not breach the brokerage agreement.
- **B.** $25,000, or the commission equivalent of 5 percent on the sale of the property for $500,000, because all conditions precedent to the nephew's duty to pay the commission were substantially fulfilled when the nephew and the developer entered into the land sale contract.
- **C.** $10,000, as a partial commission due to the agent's work done before the sale was cancelled.
- **D.** $12,500, or half the commission, because the sale reached the halfway point when the nephew and the developer signed the contract.
- **E.** Nothing, because as a third-party beneficiary of the contract between the nephew and the developer, the agent can enforce the contract only against the developer, but not against the nephew.
- **F.** $25,000, or the commission equivalent of 5 percent on the sale of the property for$500,000, because fulfillment of the consummation condition was prevented by an event beyond the agent's control.
- **G.** $25,000, because the agent provided a buyer and thus fulfilled his part of the contract.
- **H.** Nothing, because the sale did not go through and hence no commission is due.
- **I.** Nothing, because the consummation condition has not been fulfilled, and the nephew's refusal to sue the developer will not excuse that condition.
- **J.** $25,000, because the agent upheld his end of the contract in making his best efforts to find a buyer for the parcel.

**Answer: I**

*Source: mmlu_pro_law | Category: humanities | ID: humanities_002*

---

## Q123. This question refers to the following information.
"The conscience of the people, in a time of grave national problems, has called into being a new party, born of the nation's sense of justice. We of the Progressive party here dedicate ourselves to the fulfillment of the duty laid upon us by our fathers to maintain the government of the people, by the people and for the people whose foundations they laid. We hold with Thomas Jefferson and Abraham Lincoln that the people are the masters of their Constitution, to fulfill its purposes and to safeguard it from those who, by perversion of its intent, would convert it into an instrument of injustice. In accordance with the needs of each generation the people must use their sovereign powers to establish and maintain equal opportunity and industrial justice, to secure which this Government was founded and without which no republic can endure.
"This country belongs to the people who inhabit it. Its resources, its business, its institutions and its laws should be utilized, maintained or altered in whatever manner will best promote the general interest. It is time to set the public welfare in the first place."
Progressive Party Platform, 1912
"Muckraking" author Jacob A. Riis's How the Other Half Lives best exemplifies which of the following quotes from the excerpt above?

- **A.** the nation's sense of justice
- **B.** without which no republic can endure
- **C.** establish and maintain equal opportunity and industrial justice
- **D.** the people are the masters of their Constitution
- **E.** Its resources, its business, its institutions and its laws should be utilized, maintained or altered
- **F.** masters of their Constitution
- **G.** set the public welfare in the first place
- **H.** the people must use their sovereign powers
- **I.** an instrument of injustice
- **J.** the duty laid upon us by our fathers

**Answer: I**

*Source: mmlu_pro_history | Category: humanities | ID: humanities_003*

---

## Q124. This question refers to the following information.
"Those whose condition is such that their function is the use of their bodies and nothing better can be expected of them, those, I say, are slaves of nature. It is better for them to be ruled thus."
Juan de Sepulveda, Politics, 1522
"When Latin American nations gained independence in the 19th century, those two strains converged, and merged with an older, more universalist, natural law tradition. The result was a distinctively Latin American form of rights discourse. Paolo Carozza traces the roots of that discourse to a distinctive application, and extension, of Thomistic moral philosophy to the injustices of Spanish conquests in the New World. The key figure in that development seems to have been Bartolomé de Las Casas, a 16th-century Spanish bishop who condemned slavery and championed the cause of Indians on the basis of a natural right to liberty grounded in their membership in a single common humanity. 'All the peoples of the world are humans,' Las Casas wrote, and 'all the races of humankind are one.' According to Brian Tierney, Las Casas and other Spanish Dominican philosophers laid the groundwork for a doctrine of natural rights that was independent of religious revelation 'by drawing on a juridical tradition that derived natural rights and natural law from human rationality and free will, and by appealing to Aristotelian philosophy.'"
Mary Ann Glendon, "The Forgotten Crucible: The Latin American Influence on the Universal Human Rights Idea,” 2003
Which one of the following statements about the Spanish conquest of the Americas is most accurate?

- **A.** African slavery was a direct result of Spanish settlements in Florida.
- **B.** The Spanish conquest of the Americas was motivated by a desire to spread Aristotelian philosophy.
- **C.** Due to racial prejudice, Spanish explorers shunned intermarriage with native people.
- **D.** The Spanish conquest of the Americas was universally condemned by the Catholic Church.
- **E.** Juan de Sepulveda was a major critic of the Spanish conquest, due to his belief in natural law.
- **F.** Bartolomé de Las Casas supported the Spanish conquest because he believed it would improve the lives of the indigenous people.
- **G.** Early native civilizations in Mexico introduced Spanish explorers to cattle ranching and wheat cultivation.
- **H.** The Spanish conquest of the Americas led directly to the abolition of slavery.
- **I.** Christopher Columbus was not the first European to have explored North America.
- **J.** Spanish conquerors were influenced by the Native American belief in natural rights.

**Answer: I**

*Source: mmlu_pro_history | Category: humanities | ID: humanities_004*

---

## Q125. This question refers to the following information.
"When we were kids the United States was the wealthiest and strongest country in the world; the only one with the atom bomb, the least scarred by modern war, an initiator of the United Nations that we thought would distribute Western influence throughout the world. Freedom and equality for each individual, government of, by, and for the people—these American values we found good, principles by which we could live as men. Many of us began maturing in complacency.
"As we grew, however, our comfort was penetrated by events too troubling to dismiss. First, the permeating and victimizing fact of human degradation, symbolized by the Southern struggle against racial bigotry, compelled most of us from silence to activism. Second, the enclosing fact of the Cold War, symbolized by the presence of the Bomb, brought awareness that we ourselves, and our friends, and millions of abstract 'others' we knew more directly because of our common peril, might die at any time. . . ."
—Port Huron Statement, 1962
Through the remainder of the 1960s, the growth of the organization that published the Port Huron Statement can best be understood in the context of

- **A.** the increase in immigration, the growth of labor unions, and the rise of the feminist movement.
- **B.** the baby boom, economic growth, and a rapid expansion of higher education.
- **C.** the proliferation of personal computer technologies, the rise of Christian fundamentalism, and an increase in student apathy.
- **D.** the escalation of the Vietnam War, the growth of the peace movement, and the assassination of key political figures.
- **E.** economic polarization, supply-side economic policies, and the disappearance of the middle class.
- **F.** the decline of colonial powers, the rise of new independent nations, and the spread of democracy.
- **G.** the introduction of new technologies, the rise of the internet, and the growth of multinational corporations.
- **H.** the rise of the civil rights movement, the spread of communism, and the threat of nuclear war.
- **I.** the end of the Cold War, the dissolution of the Soviet Union, and the rise of globalization.
- **J.** rapid industrialization, urban growth and congestion, and corporate consolidation.

**Answer: B**

*Source: mmlu_pro_history | Category: humanities | ID: humanities_005*

---

## Q126. This question refers to the following information.
"Buckingham Palace, 10th May 1839.
The Queen forgot to ask Lord Melbourne if he thought there would be any harm in her writing to the Duke of Cambridge that she really was fearful of fatiguing herself, if she went out to a party at Gloucester House on Tuesday, an Ancient Concert on Wednesday, and a ball at Northumberland House on Thursday, considering how much she had to do these last four days. If she went to the Ancient Concert on Wednesday, having besides a concert of her own here on Monday, it would be four nights of fatigue, really exhausted as the Queen is.
But if Lord Melbourne thinks that as there are only to be English singers at the Ancient Concert, she ought to go, she could go there for one act; but she would much rather, if possible, get out of it, for it is a fatiguing time&….
As the negotiations with the Tories are quite at an end, and Lord Melbourne has been here, the Queen hopes Lord Melbourne will not object to dining with her on Sunday?"
The Letters of Queen Victoria, Volume 1 (of 3), 1837-1843: A Selection from Her Majesty's Correspondence Between the Years 1837 and 1861
The long evenings of entertainment for Queen Victoria suggest what about the nature of the English monarchy in the nineteenth century?

- **A.** That the monarchy was heavily involved in promoting the arts and music
- **B.** That true political power lay elsewhere
- **C.** That Queen Victoria had a personal preference for English singers
- **D.** That important political progress could only be made by attending social events
- **E.** That fatigue was a common issue for monarchs due to their social obligations
- **F.** That the monarchy was disconnected from the general public's day-to-day life
- **G.** That Queen Victoria was a key figure in the negotiation of political matters
- **H.** That the monarchy was primarily focused on social engagements
- **I.** That she was very fond of attending balls and concerts
- **J.** That with England's nineteenth-century economic success came more leisure time for the upper classes

**Answer: B**

*Source: mmlu_pro_history | Category: humanities | ID: humanities_006*

---

## Q127. This question refers to the following information.
"If it be conceded, as it must be by every one who is the least conversant with our institutions, that the sovereign powers delegated are divided between the General and State Governments, and that the latter hold their portion by the same tenure as the former, it would seem impossible to deny to the States the right of deciding on the infractions of their powers, and the proper remedy to be applied for their correction. The right of judging, in such cases, is an essential attribute of sovereignty, of which the States cannot be divested without losing their sovereignty itself, and being reduced to a subordinate corporate condition. In fact, to divide power, and to give to one of the parties the exclusive right of judging of the portion allotted to each, is, in reality, not to divide it at all; and to reserve such exclusive right to the General Government (it matters not by what department to be exercised), is to convert it, in fact, into a great consolidated government, with unlimited powers, and to divest the States, in reality, of all their rights, It is impossible to understand the force of terms, and to deny so plain a conclusion."
—John C. Calhoun, "South Carolina Exposition and Protest," 1828
The language of "protest" that Calhoun used in his "Exposition and Protest" was similar to the language of which of the following political positions?

- **A.** The response of the Democratic-Republicans to the Alien and Sedition Acts.
- **B.** The response of New England Federalists to the War of 1812.
- **C.** The response of the American colonists to the British Stamp Act of 1765.
- **D.** The response of the Jefferson administration to the actions of the "Barbary pirates."
- **E.** The response of the South to the Missouri Compromise of 1820.
- **F.** The response of the Confederacy to the election of Abraham Lincoln in 1860.
- **G.** The response of the North to the Fugitive Slave Act of 1850.
- **H.** The response of the Whigs to the annexation of Texas.
- **I.** The response of Daniel Shays to fiscal policies of the Massachusetts legislature in the 1780s.
- **J.** The response of supporters of Andrew Jackson to the "corrupt bargain" of 1824.

**Answer: B**

*Source: mmlu_pro_history | Category: humanities | ID: humanities_007*

---

## Q128. A man decided to stop at a drive-through hamburger stand for a late snack. As he drove up to the drive- through line, the manager of the hamburger stand informed him through the intercom system that the restaurant was closing and no further orders would be accepted. She told the man that the last car to be served was the one directly in front of him. The man became angry and yelled into the intercom machine, "Listen, babe, I am hungry. I want two cheeseburgers, a large order of fries, and a Coke. " The manager retorted, "I'm terribly sorry, but we cannot accept your order. "Shortly thereafter, the manager handed the food order to the passengers in the car immediately in front of the man's. When the man saw the manager serving that car, he became very angry, drove his automobile up to the service window and shouted at the manager, "You can't do this to me. " When the manager laughed, the man suddenly reached into the car's glove compartment and pulled out a gun. He aimed at the manager and fired the weapon, intending to hit her. The bullet missed the manager but hit a customer, wounding him in the head. In an action by the customer against the man for battery, the customer will be

- **A.** successful, because the man caused harm with his actions, regardless of his intent.
- **B.** unsuccessful, because the man was not aiming at the customer.
- **C.** unsuccessful, because the manager could have prevented the situation by accepting the man's order.
- **D.** successful, because the man was acting recklessly and created a dangerous situation.
- **E.** unsuccessful, because the man could not foresee that the bullet would hit anyone other than the manager.
- **F.** unsuccessful, because the man was provoked by the manager's refusal to serve him.
- **G.** unsuccessful, because the man did not intend to shoot the customer.
- **H.** successful, because there was a "substantial certainty" that the customer would be hit by the bullet.
- **I.** successful, because the man intended to shoot the manager.
- **J.** successful, because the bullet from the man's gun directly caused the customer's injury.

**Answer: I**

*Source: mmlu_pro_law | Category: humanities | ID: humanities_008*

---

## Q129. A lumber supplier and a fence company signed the following agreement on May 1:"The supplier promises to sell and the fence company promises to buy 7,000 sections of redwood stockade fence at $30 per section. Each section is to be made of good quality split redwood poles and is to be 7 feet long and 6 feet high; 1,000 sections are to be delivered by seller on or before June 1, and 1,000 sections by the first day in each of the following six months. Payment for the sections to be made within 10 days of delivery. "The first shipment of 1,000 sections arrived on May 27, and the fence company sent its payment on June5. The second shipment arrived on July 1, and the fence company made payment on July 5. The August shipment arrived on the afternoon of August 1. After the initial inspection, the redwood poles were found to be 7 feet long and 6. 25 feet high. The manager of the fence company then called the president of the lumber supplier. During their conversation, the president told the manager that the lumber supplier could not replace the August shipment but would allow a price adjustment. The manager refused the president's offer. The next day, the manager sent the president a fax stating that he was hereby canceling all future deliveries and returning the last shipment because of nonconformity. If the lumber supplier sues the fence company for breach of contract, the court will most likely hold that the lumber company will

- **A.** not succeed, because the fence company has the right to refuse nonconforming goods.
- **B.** not succeed, because the fence company has the right to cancel due to noncompliance with the agreed specifications.
- **C.** succeed, because the agreement did not specify that the fence company could cancel for nonconformity.
- **D.** succeed, because all deliveries to date have been timely.
- **E.** not succeed, because the deviation impaired the value of the entire contract.
- **F.** succeed, because the president offered to adjust the price for the August shipment.
- **G.** succeed, because the difference in pole height did not significantly alter the value of the fence sections.
- **H.** not succeed, because the president refused to replace the nonconforming poles.
- **I.** succeed, because the fence company did not provide adequate notice of cancellation.
- **J.** not succeed, because the fence company made all payments promptly as agreed.

**Answer: F**

*Source: mmlu_pro_law | Category: humanities | ID: humanities_009*

---

## Q130. A baseball fan purchased two tickets for a World Series baseball game. The fan contacted his best friend and invited him to go to the game. The friend, who was a fanatic baseball fan, eagerly agreed. The fan told the friend that the game started at 7:00 p. m. and that he would pick him up at about 5:00 p. m. so they could get there early to watch batting practice. They were driving to the game together when the fan sped up to cross an intersection while the traffic signal was changing from amber to red. As he reached the intersection, the fan was traveling at 50 m. p. h. although the posted speed limit was 25 m. p. h. Simultaneously, a car entered the intersection on red and collided with the fan's vehicle. The friend suffered a broken pelvis in the collision. This jurisdiction has adopted the following "modified" comparative negligence statute:"A negligent plaintiff is entitled to obtain a recovery provided plaintiff's negligence is not equal to or greater than that of the defendant's; otherwise no recovery is permitted. "Suppose the friend brings suit against the driver of the car that entered the intersection on the red light to recover damages for his injury. Ajury returned a special verdict with the following findings: (1) The fan was 55 percent negligent in speeding; (2) The driver was 45 percent negligent in driving through the red light; and (3) The friend suffered $100,000 in damages. As a result, the court should enter a judgment for the friend in the amount of

- **A.** $45,000. 00
- **B.** $50,000.00
- **C.** nothing, because the fan was more negligentthan the driver.
- **D.** A split judgment where the fan pays $55,000.00 and the driver pays $45,000.00
- **E.** $100,000.00 but paid by both the fan and the driver
- **F.** $45,000.00 but only from the driver's insurance
- **G.** $55,000. 00
- **H.** $55,000.00 but only from the fan's insurance
- **I.** $55,000.00 from the driver as the fan was more negligent
- **J.** $100,000. 00

**Answer: J**

*Source: mmlu_pro_law | Category: humanities | ID: humanities_010*

---

## Q131. In 1981, a devoted conservationist, was the owner of a 100-acre tract of undeveloped land. In that year, the conservationist conveyed the tract "to my nephew in fee simple, provided, however, that the grantee agrees that neither he nor his heirs or assigns shall ever use the property for any commercial purpose. If any portion of said tract is used for other than residential purposes, then the grantor or his successors in interest may re-enter as of the grantor's former estate. " This deed was properly recorded. The nephew died intestate in 1999, survived by his wife. The conservationist died in 2002, survived by his two daughters, his only heirs. During the period between 1981 and 2007, the spreading development from a nearby city began to engulf the tract. Though still undeveloped, the tract became surrounded by office buildings, shopping malls, and other commercial edifices. In 2009, the wife executed and delivered to a developer a fee simple conveyance of the tract, which the developer immediately recorded. The deed did not contain any reference to the restriction noted above. After the developer acquired title to the tract, he commenced construction of a hotel complex on a portion of the tract that bordered an apartment building. The applicable recording statute in effect in this jurisdiction provides, in part, "No deed or other instrument in writing, not recorded in accordance with this statute, shall affect the title or rights to, in any real estate, or any devisee or purchaser in good faith, without knowledge of the existence of such unrecorded instruments. "If one of the daughters brings suit to enjoin the developer from constructing the hotel, the plaintiff will most likely

- **A.** lose, because the developer was a bona fide purchaser for value without notice of the restriction.
- **B.** win, because either daughter has the right of re-entry for condition broken.
- **C.** win, because the restriction on commercial use was recorded and thus the developer should have been aware of it.
- **D.** lose, because a common development scheme had been established for the entire tract.
- **E.** win, because either daughter's right to the tract vested immediately upon the developer's construction of the hotel complex.
- **F.** lose, because the restriction was not included in the deed from the wife to the developer.
- **G.** lose, because the wife had the right to sell the property without any restrictions.
- **H.** win, because the developer violated the deed's prohibition against commercial use.
- **I.** win, because the daughters, as the conservationist's only heirs, received a valid possibility of reverter from their father.
- **J.** lose, because the restriction on the use of the property is unenforceable.

**Answer: B**

*Source: mmlu_pro_law | Category: humanities | ID: humanities_011*

---

## Q132. This question refers to the following information.
Read the following governmental report.
Of the 450 sick persons whom the inhabitants were unable to relieve, 200 were turned out, and these we saw die one by one as they lay on the roadside. A large number still remain, and to each of them it is only possible to dole out the least scrap of bread. We only give bread to those who would otherwise die. The staple dish here consists of mice, which the inhabitants hunt, so desperate are they from hunger. They devour roots which the animals cannot eat; one can, in fact, not put into words the things one sees. . . . This narrative, far from exaggerating, rather understates the horror of the case, for it does not record the hundredth part of the misery in this district. Those who have not witnessed it with their own eyes cannot imagine how great it is. Not a day passes but at least 200 people die of famine in the two provinces. We certify to having ourselves seen herds, not of cattle, but of men and women, wandering about the fields between Rheims and Rhétel, turning up the earth like pigs to find a few roots; and as they can only find rotten ones, and not half enough of them, they become so weak that they have not strength left to seek food. The parish priest at Boult, whose letter we enclose, tells us he has buried three of his parishioners who died of hunger. The rest subsisted on chopped straw mixed with earth, of which they composed a food which cannot be called bread. Other persons in the same place lived on the bodies of animals which had died of disease, and which the curé, otherwise unable to help his people, allowed them to roast at the presbytery fire.
—Report of the Estates of Normandy, 1651
Which of the following contributed the LEAST to the health and hunger problems faced by the French people in the seventeenth century?

- **A.** Low taxes on the peasants and middle class
- **B.** War and conflict
- **C.** Low-productivity agricultural practices
- **D.** Lack of farming tools
- **E.** Inadequate food storage facilities
- **F.** Adverse weather
- **G.** High taxes on the nobility and clergy
- **H.** Poor transportation
- **I.** Overpopulation
- **J.** Lack of medical knowledge

**Answer: A**

*Source: mmlu_pro_history | Category: humanities | ID: humanities_012*

---

## Q133. A doctor was the owner of 1,500 acres of undeveloped timberland. In September 1989, the doctor executed a warranty deed conveying the timberland property to a dentist in fee simple. The dentist recorded immediately in the Grantor Grantee Index. Then in April 1990, the dentist conveyed the same tract to a buyer in fee simple by warranty deed. The buyer paid full market value and recorded the deed at once in the Grantor Grantee Index. The land in question had never been occupied, fenced, or cleared except that between the years 1986 2010, a mining company, one mile from the property, regularly drove trucks over a cleared path pursuant to a 1986 agreement with the doctor. The agreement, which was duly recorded, provided that "the parties expressly agree and the doctor promises that the doctor and his successors shall refrain from obstructing the said described pathway across the doctor's land, which the mining company and its successors may perpetually use as a road, in consideration of which the mining company and its successors will pay the sum of $700 per annum. "In 1990, after the conveyance from the dentist, the buyer informed the mining company that he would no longer honor the 1986 agreement permitting the mining company to use the pathway. The mining company brought an action for specific performance. Judgment should be for

- **A.** the buyer, because the mining company's use of the pathway was not a legally binding agreement.
- **B.** the mining company, because the possessor of a servient interest would prevail against subsequent owners.
- **C.** the mining company, because the agreement gives them perpetual use of the pathway.
- **D.** the mining company, because the agreement was duly recorded and thus legally binding.
- **E.** the buyer, because the mining company has no legal right to the use of the pathway.
- **F.** the buyer, because there was no privity of estate between the buyer and the mining company.
- **G.** the buyer, because the agreement was with the original landowner, not the buyer.
- **H.** the mining company, because they have paid an annual fee for the use of the pathway.
- **I.** the mining company, because their property interest would "run with the land. "
- **J.** the buyer, because the mining company's interest was extinguished by the subsequent conveyance.

**Answer: I**

*Source: mmlu_pro_law | Category: humanities | ID: humanities_013*

---

## Q134. This question refers to the following information.
"Wherever I go—the street, the shop, the house, or the steamboat—I hear the people talk in such a way as to indicate that they are yet unable to conceive of the Negro as possessing any rights at all. Men who are honorable in their dealings with their white neighbors will cheat a Negro without feeling a single twinge of their honor. To kill a Negro they do not deem murder; to debauch a Negro woman they do not think fornication; to take the property away from a Negro they do not consider robbery. The people boast that when they get freedmen affairs in their own hands, to use their own classic expression, 'the niggers will catch hell.'
"The reason of all this is simple and manifest. The whites esteem the blacks their property by natural right, and however much they may admit that the individual relations of masters and slaves have been destroyed by the war and the President's emancipation proclamation, they still have an ingrained feeling that the blacks at large belong to the whites at large, and whenever opportunity serves they treat the colored people just as their profit, caprice or passion may dictate."
—Congressional testimony of Col. Samuel Thomas, Assistant Commissioner, Bureau of Refugees, Freedmen and Abandoned Lands, 1865
To address the problems identified in Federalist #15, Hamilton proposed

- **A.** Banning the slave trade to promote equality among all races.
- **B.** Implementing a system of federal taxation to boost the national economy.
- **C.** Encouraging industrialization and urbanization to improve living standards.
- **D.** Implementing a national education system to improve literacy rates.
- **E.** forging alliances with American Indian nations to present a united front to European powers.
- **F.** Establishing a national bank to control the country's finances.
- **G.** Introducing a bill of rights to protect individual liberties.
- **H.** increasing spending on military forces and cutting spending on social programs.
- **I.** adopting a new constitution in order to create a more national government.
- **J.** abandoning an isolationist approach to foreign policy and adopting a more aggressive and interventionist stance.

**Answer: I**

*Source: mmlu_pro_history | Category: humanities | ID: humanities_014*

---

## Q135. This question refers to the following information.
"In 1500 that work appeared which Erasmus had written after his misfortune at Dover, and had dedicated to Mountjoy, the Adagiorum Collectanea. It was a collection of about eight hundred proverbial sayings drawn from the Latin authors of antiquity and elucidated for the use of those who aspired to write an elegant Latin style. In the dedication Erasmus pointed out the profit an author may derive, both in ornamenting his style and in strengthening his argumentation, from having at his disposal a good supply of sentences hallowed by their antiquity. He proposes to offer such a help to his readers. What he actually gave was much more. He familiarized a much wider circle than the earlier humanists had reached with the spirit of antiquity.
Until this time the humanists had, to some extent, monopolized the treasures of classic culture, in order to parade their knowledge of which the multitude remained destitute, and so to become strange prodigies of learning and elegance. With his irresistible need of teaching and his sincere love for humanity and its general culture, Erasmus introduced the classic spirit, in so far as it could be reflected in the soul of a sixteenth-century Christian, among the people. Not he alone; but none more extensively and more effectively. Not among all the people, it is true, for by writing in Latin he limited his direct influence to the educated classes, which in those days were the upper classes.
Erasmus made current the classic spirit. Humanism ceased to be the exclusive privilege of a few. According to Beatus Rhenanus he had been reproached by some humanists, when about to publish the Adagia, for divulging the mysteries of their craft. But he desired that the book of antiquity should be open to all."
Johan Huizinga, twentieth-century Dutch philosopher, Erasmus and the Age of Reformation, 1924
The type of humanism attributed to Erasmus in this passage is most similar to what Southern Renaissance movement?

- **A.** Protestant Reformation
- **B.** Existentialism
- **C.** Naturalism
- **D.** Empiricism
- **E.** Antitrinitarianism
- **F.** Neoplatonism
- **G.** Pragmatism
- **H.** Stoicism
- **I.** Pietism
- **J.** Rationalism

**Answer: F**

*Source: mmlu_pro_history | Category: humanities | ID: humanities_015*

---

## Q136. This question refers to the following information.
The text below is the government proclamation.
On the basis of the above-mentioned new arrangements, the serfs will receive in time the full rights of free rural inhabitants.
The nobles, while retaining their property rights to all the lands belonging to them, grant the peasants perpetual use of their household plots in return for a specified obligation[; . . . the nobles] grant them a portion of arable land fixed by the said arrangements as well as other property. . . . While enjoying these land allotments, the peasants are obliged, in return, to fulfill obligations to the noblemen fixed by the same arrangements. In this status, which is temporary, the peasants are temporarily bound. . . .
[T]hey are granted the right to purchase their household plots, and, with the consent of the nobles, they may acquire in full ownership the arable lands and other properties which are allotted them for permanent use. Following such acquisition of full ownership of land, the peasants will be freed from their obligations to the nobles for the land thus purchased and will become free peasant landowners.
WE have deemed it advisable:
3. To organize Peace Offices on the estates of the nobles, leaving the village communes as they are, and to open cantonal offices in the large villages and unite small village communes.
4. To formulate, verify, and confirm in each village commune or estate a charter which will specify, on the basis of local conditions, the amount of land allotted to the peasants for permanent use, and the scope of their obligations to the nobleman for the land.
6. Until that time, peasants and household serfs must be obedient towards their nobles, and scrupulously fulfill their former obligations.
7. The nobles will continue to keep order on their estates, with the right of jurisdiction and of police, until the organization of cantons and of cantonal courts.
—Alexander II, "The Abolition of Serfdom in Russia," Manifesto of February 19, 1861
Which of the following was a major impetus in convincing Tsar Alexander II of the necessity of freeing the serfs?

- **A.** Recent defeat in the Crimean War convinced the tsar some domestic reforms were necessary.
- **B.** The Tsar wanted to improve his popularity among the Russian people.
- **C.** The spread of socialist ideas among the serfs was causing unrest.
- **D.** The increasing population of serfs was becoming too difficult to manage.
- **E.** A labor force to complete the Trans-Siberian Railroad was needed as well as military recruits.
- **F.** The Tsar was motivated by a desire to modernize and industrialize Russia.
- **G.** The Decembrist Revolt and its aftermath had convinced the young tsar to make reforms.
- **H.** The Tsar believed that freeing the serfs would help Russia in its competition with Western powers.
- **I.** The Tsar was influenced by the writings of liberal philosophers.
- **J.** Enlightened rulers in Prussia and Austria had recently done the same, which pressured Alexander II to act.

**Answer: A**

*Source: mmlu_pro_history | Category: humanities | ID: humanities_016*

---

## Q137. A rancher is the owner of a ranch that is situated upon the top of a mountain. Located below the ranch is an estate that is owned by a millionaire. A stream is a non-navigable watercourse that originates at the top of the mountain and runs all the way down into a valley. Both the ranch and the estate are within the watershed of the stream. When the millionaire purchased the estate in 1956, he started taking water from the stream and used it to irrigate the southern half of his property, which he has used as a farm. Prior to 1956, the southern half of the estate had been cleared and placed in cultivation, while the northern half remained wooded and virtually unused. The millionaire continued this established pattern of use and has never stopped using the water in this way. In 1986, the rancher built a home on the ranch and started talcing water from the stream for domestic purposes. During that year there was heavy rainfall, and this caused the stream to run down the mountain at a high water level. However, in 1987, a drought caused the stream to flow at a very low level. Consequently, there was only enough water to irrigate the millionaire's farmland or, in the alternative, to supply all of the rancher's domestic water needs and one-quarter of the millionaire's irrigation requirements. The mountain is located in a jurisdiction where the period of prescription is 15 years. The rancher is continuing to take water for his personal needs and there is insufficient water to irrigate the estate. The millionaire then brings an appropriate action in 1996 to declare that his water rights to the stream are superior to those of the rancher. In addition, the millionaire moves to have the full flow of the stream passed to him, notwithstanding the effect it might have on the rancher. If this state follows the common law of riparian rights, but does not follow the doctrine of prior appropriation, judgment should be for whom?

- **A.** The rancher, because the drought conditions give him priority access to the water.
- **B.** The millionaire, as he was using the water first for his estate's needs.
- **C.** The millionaire, because he has a right to the water for irrigation purposes.
- **D.** Neither, because both have equal rights to the water as it runs through both of their properties.
- **E.** The millionaire, because he put the water to a beneficial use prior to the rancher's use and has continuously used the water.
- **F.** The rancher, because the millionaire's use of the water is excessive and unnecessary.
- **G.** The millionaire, because he obtained an easement by prescription to remove as much water as he may need.
- **H.** The rancher, because domestic use is superior to and is protected against an agricultural use.
- **I.** The rancher, because as an upstream landowner, he would have superior rights to the water than a downstream owner.
- **J.** The millionaire, because he has been using the water for a longer period of time.

**Answer: H**

*Source: mmlu_pro_law | Category: humanities | ID: humanities_017*

---

## Q138. Ann, Bea, and Carla were three friends who lived in the same neighborhood. While Ann was away on a business trip, someone broke into her garage and stole her golf clubs. The next week, Ann was planning to go on vacation and asked Bea if she could borrow her golf clubs. Bea agreed and loaned her golf clubs to Ann, who promised to return them after her vacation. When Ann returned home, she kept the golf clubs and continued to use them. A few weeks later, Bea was having dinner with Carla and learned that Carla owed Ann $4,000. Carla had just been laid off from her job and did not have the money to repay Ann. Bea told Carla that she would contact Ann and make arrangements to repay the loan on her behalf. Thereupon, Ann and Bea entered into a written agreement wherein Bea promised to pay Ann, at a rate of $400 a month, the matured $4,000 debt that Carla owed Ann. In the same written instrument, Ann promised to return Bea's golf clubs, which she still had in her possession. Ann, however, made no written or oral. commitment to forbear to sue Carla to collect the $4,000 debt; and Bea made no oral or written request for any such forbearance. After this agreement between Ann and Bea was signed and executed, Ann promptly returned the golf clubs to Bea. For the next six months, Bea made and Ann accepted the $400 monthly payments as agreed. During that period, Ann, in fact, did forbear to take any legal action against Carla. However, Bea then repudiated her agreement with Ann, and 30 days later Ann filed a contract action against Bea. Assume that the applicable statute of limitations on Ann's antecedent claim against Carla expired the day before Ann filed her contract action against Bea. Which of the following is the most persuasive argument that Bea is not liable to Ann under the terms of their written agreement?

- **A.** Since the agreement did not specify the consequences if Bea failed to make the payments, Bea is not liable to Ann.
- **B.** Since Bea had already begun making payments to Ann, there was no valid contract between them.
- **C.** Since the written agreement between Bea and Ann shows a gross imbalance between the values of the promises exchanged, the consideration for Bea's promise was legally insufficient to support it.
- **D.** Since Carla, when the agreement between Ann and Bea was made, had a pre-existing duty to repay the $4,000 debt to Ann, there was no consideration for Bea's promise to Ann.
- **E.** Since Ann did not expressly promise to forbear to sue Carla to collect the antecedent $4,000 debt, Ann's forbearance for six months could not constitute consideration for Bea's promise.
- **F.** Since Ann did not take any legal action against Carla, Bea's promise to repay the debt was unnecessary and thus lacks consideration.
- **G.** Since Ann returned Bea's golf clubs before the agreement was signed, Bea's promise to repay the $4,000 debt lacks consideration.
- **H.** Since Bea had made no oral or written request for Ann's forbearance, Bea's promise to repay the debt lacks consideration.
- **I.** Since Ann had a pre-existing duty to return Bea's golf clubs to her when the agreement between Ann and Bea was made, there was no consideration for Bea's promise to Ann.
- **J.** Since the statute of limitations on Ann's claim against Carla had expired, Bea's promise to repay the debt is not enforceable.

**Answer: I**

*Source: mmlu_pro_law | Category: humanities | ID: humanities_018*

---

## Q139. The police received an anonymous tip informing them that a pharmacist was engaged in the illegal manufacture of synthetic cocaine. As part of its investigation, the police placed an electronic tracking device on the pharmacist's car. The tracking device was attached to the underbody of the pharmacist's car while it was parked outside his home. The police did not secure a warrant before installing the device. By means of the tracking device, the police were able to trail the pharmacist's movements. The police followed the pharmacist every day for almost a month. Finally, one day the police tracked the pharmacist's car to a vacant warehouse on the outskirts of town. While the pharmacist was inside the building, the police peered in the window and saw drug paraphernalia and equipment used in the manufacture of synthetic cocaine. Based on these observations, the police secured a search warrant and gained entry into the building. Once inside, the police arrested the pharmacist and confiscated a large quantity of synthetic cocaine that had just been produced. At his trial for illegal possession and manufacture of a controlled dangerous substance, the pharmacist moves to suppress the cocaine confiscated by the police. The pharmacist's motion will most likely be

- **A.** denied, because the police acted in good faith when placing the tracking device.
- **B.** denied, because the police had reasonable suspicion to track the pharmacist's movements.
- **C.** granted, because the pharmacist had a reasonable expectation of privacy in his car.
- **D.** granted, because the police did not have a warrant to place the tracking device on the pharmacist's car.
- **E.** denied, because the police could have discovered the location of the warehouse simply by following the pharmacist's car.
- **F.** denied, because the evidence would have inevitably been discovered.
- **G.** denied, because the electronic surveillance of the pharmacist's car did not exceed 30 days.
- **H.** granted, because the police invaded the pharmacist's privacy by peering into the warehouse window.
- **I.** granted, because the information upon which the search warrant was based was illegally obtained by means of the tracking device.
- **J.** granted, because the seizure must be suppressed as the fruit of an illegal search.

**Answer: E**

*Source: mmlu_pro_law | Category: humanities | ID: humanities_019*

---

## Q140. This question refers to the following information.
Read the following memoir.
Not only did he expect all persons of distinction to be in continual attendance at Court, but he was quick to notice the absence of those of inferior degree; at his lever, his couches, his meals, in the gardens of Versailles (the only place where the courtiers in general were allowed to follow him), he used to cast his eyes to right and left; nothing escaped him[;] he saw everybody. If anyone habitually living at Court absented himself he insisted on knowing the reason; those who came there only for flying visits had also to give a satisfactory explanation; anyone who seldom or never appeared there was certain to incur his displeasure. If asked to bestow a favor on such persons he would reply haughtily: "I do not know him"; of such as rarely presented themselves he would say, "He is a man I never see"; and from these judgments there was no appeal.
No one understood better than Louis XIV the art of enhancing the value of a favor by his manner of bestowing it; he knew how to make the most of a word, a smile, even of a glance.
He loved splendor, magnificence, and profusion in all things, and encouraged similar tastes in his Court; to spend money freely on equipages and buildings, on feasting and at cards, was a sure way to gain his favor, perhaps to obtain the honor of a word from him. Motives of policy had something to do with this; by making expensive habits the fashion, and, for people in a certain position, a necessity, he compelled his courtiers to live beyond their income, and gradually reduced them to depend on his bounty for the means of subsistence.
—Duke Saint-Simon, Memoirs of Louis XIV and His Court and His Regency, c. 1750
Which of the following was the greatest weakness and regret of the rule of King Louis XIV?

- **A.** His inability to produce a male heir led to a succession crisis.
- **B.** His insistence on religious uniformity led to civil unrest and division.
- **C.** He was so concerned with ceremonies and appearances that he did not rule his country well.
- **D.** His domination of the nobility left him without friends and allies.
- **E.** He left the administration of his kingdom to professional bureaucrats known as intendants.
- **F.** His lavish spending led to the financial ruin of his kingdom.
- **G.** He was too focused on architectural projects, neglecting the needs of his people.
- **H.** He was at war for 2/3 of his reign and united the other major powers against him.
- **I.** He failed to modernize France's military, leaving it vulnerable to foreign attacks.
- **J.** His lack of interest in foreign affairs led to international isolation.

**Answer: H**

*Source: mmlu_pro_history | Category: humanities | ID: humanities_020*

---

## Q141. In 1996, a developer purchased a 100-acre tract located in a northern county in a state. Shortly thereafter, the developer prepared a subdivision plan that created 100 one-acre residential building lots on this tract. In 1997, the subdivision plan was recorded with the county recorder's office. During the next few years, the developer sold 60 residential lots to individual purchasers. Each deed specified that every lot designated on the subdivision plan was to be recorded in the county recorder's office. Each deed also provided the following:"No house trailer or mobile home shall be built or maintained on any lot within the subdivision. "In 2003, the developer conveyed the remaining 40 lots to a builder by deed that included language identical to that contained in the first 60 deeds. This deed from the developer to the builder was recorded. By 2008, the builder had sold all of the 40 lots. Each of these deeds identified each lot as being a part of the subdivision, but did not include the clause relating to mobile homes. On January 30, 2009, a buyer, who had purchased one of the residential lots from the builder, placed a mobile home on his property. Which of the following statements is LEAST accurate with respect to the buyer's deed?

- **A.** The buyer has no obligation to remove the mobile home.
- **B.** All subsequent grantees of the builder would be in privity of contract.
- **C.** The deed from the builder to the buyer did not include the covenant prohibiting mobile homes.
- **D.** The covenant prohibiting mobile homes ran with the land as far as the builder, but not as far as the buyer.
- **E.** All subsequent grantees of the builder would be in privity of estate.
- **F.** The buyer should have had constructive notice of the restriction against mobile homes.
- **G.** The buyer should have had actual notice of the restriction against mobile homes.
- **H.** The covenant prohibiting mobile homes could be enforced by any subdivision lot owner.
- **I.** The covenant prohibiting mobile homes was not recorded with the county recorder's office.
- **J.** The covenant prohibiting mobile homes could only be enforced by the original developer.

**Answer: D**

*Source: mmlu_pro_law | Category: humanities | ID: humanities_021*

---

## Q142. A football player was the star fulllack for the local college football team. After missing two practices, the football player was dropped from the team by the head football coach. Following his dismissal, the football player met with the coach and asked if he could rejoin the team. The coach said that the football player was despised by the other players and under no circumstances could he return to the team. As the football player was leaving the coach's office, feeling very dejected, the coach then said to him, "Hope you decide to transfer, because everybody hates your guts around here. "Later that same evening, the football player wrote a suicide note in which he stated, "The coach is responsible for my despondency. If I can't play football, I don't want to live. " After swallowing a bottle of Quaalude barbiturates, the football player fell unconscious in his dormitory room. Moments later, the football player's roommate entered the room and saw his limp body on the floor. The roommate read the suicide note and then attempted to administer aid. Failing to revive him, the roommate picked up the football player and carried him to the college's first aid center. The football player received prompt medical attention and soon recovered from his drug overdose. If the football player asserts a claim against the coach based on intentional infliction of emotional distress, the football player will most likely

- **A.** not prevail, because the coach had the right to express his opinion.
- **B.** prevail, because the coach's remarks led directly to the football player's emotional distress and subsequent suicide attempt.
- **C.** prevail, because the coach intended to cause him to suffer emotional distress.
- **D.** not prevail, because the football player's drug overdose resulted from his own voluntary act.
- **E.** prevail, because the coach's remarks constituted bullying behavior.
- **F.** prevail, because the coach's remarks were intended to inflict emotional distress and resulted in the football player's suicide attempt.
- **G.** not prevail, because the coach acted reasonably under the circumstances, since everyone on the team hated the football player.
- **H.** prevail, because the coach's remark did, in fact, cause the football player to suffer emotional distress.
- **I.** not prevail, because the coach's remarks do not meet the legal standard for intentional infliction of emotional distress.
- **J.** not prevail, because the football player's overdose was an unforeseen consequence of the coach's remarks.

**Answer: D**

*Source: mmlu_pro_law | Category: humanities | ID: humanities_022*

---

## Q143. An environmentalist was very interested in environmental issues, particularly protection of wetland areas. He decided to dig out the lawn in his back yard and turn the space into a swampy marsh. Eventually, his back yard was filled with tall grasses, reeds, and other marsh plants. A wide variety of frogs, turtles, snakes, birds, and other animals inhabited the yard. The ground was usually covered by several inches of standing water. The environmentalist's neighbors were not pleased with the condition of the environmentalist's yard. They complained that it produced foul odors, and they claimed that the standing water was a breeding ground for mosquitoes and other insects. Several months after the environmentalist converted his yard into a marsh, a real estate investor purchased the house closest to the environmentalist's back yard swamp. The investor lived in a large city several hundred miles away, and he purchased the house next to the environmentalist's for investment purposes. The investor rented the house to a family under a long-term lease. The tenant family complained frequently to the investor about being annoyed by the environmentalist's yard. If the investor asserts a nuisance claim against the environmentalist, the environmentalist's best defense would be

- **A.** that the investor failed to conduct a proper inspection of the property and surrounding area before purchasing the house.
- **B.** that the investor owns the property but has rented it out, so the investor does not have actual possession or the right to immediate possession of the land.
- **C.** that the environmentalist's yard is actually beneficial to the community by providing a natural habitat for local wildlife.
- **D.** that the environmentalist has a right to use his property as he sees fit, as long as it does not harm others.
- **E.** that the investor has not shown that the marsh has negatively affected the value of his property.
- **F.** that the swampy condition of his yard attracts a variety of wildlife, which increases biodiversity.
- **G.** that when the investor purchased the house, he knew or should have known about the swampy condition of the environmentalist's property.
- **H.** that he had sound environmental reasons for maintaining the swampy condition of his yard.
- **I.** that the standing water in his yard is not the source of the mosquito problem.
- **J.** that turning his yard into a swampy marsh did not violate any zoning ordinance.

**Answer: B**

*Source: mmlu_pro_law | Category: humanities | ID: humanities_023*

---

## Q144. This question refers to the following information.
"The quicksilver mines of Huancavelica are where the poor Indians are so harshly punished, where they are tortured and so many Indians die; it is there that the noble caciques [headmen] of this kingdom are finished off and tortured. The same is true in all the other mines: the silver mines of Potosi [and others]….The owners and stewards of the mines, whether Spaniards, mestizos, or Indians, are such tyrants, with no fear of God or Justice, because they are not audited and are not inspected twice a year….
And they are not paid for the labor of traveling to and from the mines or for the time they spend at the mines. The Indians, under the pretext of mining chores, are made to spend their workdays herding cattle and conveying goods; they are sent off to the plains, and the Indians die. These Indians are not paid for their labor, and their work is kept hidden.
And [the mine owners] keep Indian cooking women in their residences; they use cooking as a pretext for taking concubines….And they oblige the Indians to accept corn or meat or chicha [corn beer]…at their own expense, and they deduct the price from their labor and their workdays. In this way, the Indians end up very poor and deep in debt, and they have no way to pay their tribute.
There is no remedy for all this, because any [colonial official] who enters comes to an agreement with the mine owners, and all the owners join forces in bribing him….Even the protector of the Indians is useless;…he [does not] warn Your Majesty or your royal Audiencia [court] about the harms done to the poor Indians."
Excerpt from The First New Chronicle and Good Government [abridged], by Felipe Guaman Poma de Alaya. Selected, translated, and annotated by David Frye. Copyright 2006 Hackett Publishing Company. Reprinted with permission from the publisher.
Felipe Guaman Poma de Ayala, The First New Chronicle and Good Government, ca. 1610
The production of the mines mentioned in the passage most directly contributed to which of the following in the period 1450–1750 C.E.?

- **A.** The rise of democratic institutions in Spain
- **B.** The emergence of Spain as a leading power in the arts and sciences
- **C.** A decrease in the frequency of voyages of exploration undertaken by the Spanish
- **D.** The decrease in the use of native labor in the Spanish colonies.
- **E.** A decrease in patronage of religious activities by the monarchs of Spain
- **F.** The abolition of the feudal system in Spain
- **G.** The prosecution of a variety of wars by the Spanish Hapsburgs across the world
- **H.** A decline in the influence of the Catholic Church in Spain
- **I.** The development of a vibrant merchant class in Spain
- **J.** An increase in the migration of Spanish citizens to the New World

**Answer: G**

*Source: mmlu_pro_history | Category: humanities | ID: humanities_024*

---

## Q145. This question refers to the following information.
The 1980s have been born in turmoil, strife, and change. This is a time of challenge to our interests and our values and it's a time that tests our wisdom and skills.
At this time in Iran, 50 Americans are still held captive, innocent victims of terrorism and anarchy. Also at this moment, massive Soviet troops are attempting to subjugate the fiercely independent and deeply religious people of Afghanistan. These two acts—one of international terrorism and one of military aggression—present a serious challenge to the United States of America and indeed to all the nations of the world. Together we will meet these threats to peace.…
Three basic developments have helped to shape our challenges: the steady growth and increased projection of Soviet military power beyond its own borders; the overwhelming dependence of the Western democracies on oil supplies from the Middle East; and the press of social and religious and economic and political change in the many nations of the developing world, exemplified by the revolution in Iran.
Each of these factors is important in its own right. Each interacts with the others. All must be faced together, squarely and courageously. We will face these challenges, and we will meet them with the best that is in us. And we will not fail.
—Jimmy Carter, State of the Union Address, January 23, 1980
The situation Carter described led most directly to which of the following?

- **A.** The withdrawal of Soviet troops from Afghanistan
- **B.** Carter's victory in the next presidential election
- **C.** Carter's defeat in the next presidential election
- **D.** An American invasion in the Middle East
- **E.** The establishment of a new government in Afghanistan
- **F.** An economic boom in the United States
- **G.** The creation of the North Atlantic Treaty Organization (NATO)
- **H.** A diplomatic resolution with the Soviet Union
- **I.** The establishment of the United Nations
- **J.** The signing of a peace treaty with Iran

**Answer: C**

*Source: mmlu_pro_history | Category: humanities | ID: humanities_025*

---

## Q146. This question refers to the following information.
Read the excerpts below.
This corruption is repeatedly designated by Paul by the term sin . . . such as adultery, fornication, theft, hatred, murder, revellings, he terms, in the same way, the fruits of sin, though in various passages of Scripture . . . we are, merely on account of such corruption, deservedly condemned by God, to whom nothing is acceptable but righteousness, innocence, and purity.
—John Calvin, from The Institutes of Christian Religion, Book 2: Chapter 1, 1545
The covenant of life is not preached equally to all, and among those to whom it is preached, does not always meet with the same reception. This diversity displays the unsearchable depth of the divine judgment, and is without doubt subordinate to God's purpose of eternal election. But if it is plainly owing to the mere pleasure of God that salvation is spontaneously offered to some, while others have no access to it, great and difficult questions immediately arise, questions which are inexplicable, when just views are not entertained concerning election and predestination[,] . . . the grace of God being illustrated by the contrast, viz., that he does not adopt all promiscuously to the hope of salvation, but gives to some what he denies to others.
—John Calvin, from The Institutes of Christian Religion, Book 3: Chapter 21, 1545
Which of the following justifications used by Protestant reformers such as Calvin is alluded to above?

- **A.** The belief that everyone has direct access to God, without the need for priests or church hierarchy.
- **B.** The belief in the necessity of separation of church and state.
- **C.** They believed in religious tolerance and the acceptance of different faiths.
- **D.** Religion was used to challenge the authority of earthly monarchs.
- **E.** The corruption of the Roman Catholic Church and its leaders meant that reform was needed.
- **F.** They believed that their church should not be subordinate to the state.
- **G.** The notion that salvation is predetermined and not all individuals have access to it.
- **H.** The idea that churches should be self-governed and independent.
- **I.** The idea that religious teachings should be made available in the vernacular rather than in Latin.
- **J.** The concept that salvation comes from faith alone rather than through good works is supported.

**Answer: E**

*Source: mmlu_pro_history | Category: humanities | ID: humanities_026*

---

## Q147. This question refers to the following information.
"XI. As the present sciences are useless for the discovery of effects, so the present system of logic is useless for the discovery of the sciences.
XIX. There are and can exist but two ways of investigating and discovering truth. The one hurries on rapidly from the senses and particulars to the most general axioms, and from them, as principles and their supposed indisputable truth, derives and discovers the intermediate axioms. This is the way now in use. The other constructs its axioms from the senses and particulars, by ascending continually and gradually, till it finally arrives at the most general axioms, which is the true but unattempted way.
XXII. Each of these two ways begins from the senses and particulars, and ends in the greatest generalities&…
XXXVI. We have but one simple method of delivering our sentiments, namely, we must bring men to particulars and their regular series and order, and they must for a while renounce their notions, and begin to form an acquaintance with things."
Francis Bacon, English philosopher and essayist, Novum Organum, 1620
By the 1800s, the method of empirical reasoning reflected in the passage had undergone which of the following changes?

- **A.** It had been expanded upon to include non-empirical forms of reasoning.
- **B.** It had become a core principle of European culture.
- **C.** It had stagnated to the point that the common person had begun to search for a new organizing principle of life.
- **D.** It was only used in select areas of academic study.
- **E.** It had been replaced entirely by a different method of reasoning.
- **F.** It had become so widely accepted that it was no longer questioned.
- **G.** It had weakened to the point of irrelevance.
- **H.** It had been refined and changed by so many people that it had become unrecognizable to those such as Bacon who had pioneered it.
- **I.** It had been altered to incorporate religious beliefs into the scientific process.
- **J.** It had been completely dismissed by the scientific community.

**Answer: B**

*Source: mmlu_pro_history | Category: humanities | ID: humanities_027*

---

## Q148. During spring break, a private boarding school was deserted while students and teachers were away on vacation. A guidance counselor remained on campus because he was working on a research project. After working late one night, the counselor decided to enter the room of a student from a very wealthy family. The counselor was rummaging through the student's room looking for something valuable to steal. Under the bed, he noticed an expensive suitcase. The counselor opened the suitcase and found an express mail envelope. The counselor knew that the student's father often sent money to his son in express mail envelopes. The counselor opened the envelope and saw that it contained a large quantity of white powder, which he suspected to be heroin. The counselor telephoned the police, and an officer was dispatched to the school. The counselor handed the officer the envelope, which he transported to the police station. At the station house, the officer opened the envelope and sent a sampling of the substance to the police lab. Tests confirmed the substance to be heroin. The police did not secure a search warrant before confiscating and opening the envelope. The student was thereafter arrested and charged with unlawful possession of a controlled dangerous substance. The student's attorney has filed a motion to suppress the heroin from evidence. The motion will most likely be 

- **A.** granted, because the student was not present during the search.
- **B.** denied, because the school has a policy allowing searches of student rooms.
- **C.** granted, because the police did not have probable cause to test the substance.
- **D.** granted, because the police should have secured a warrant before opening the envelope.
- **E.** denied, because the search was conducted by a private party.
- **F.** granted, because the student's room is considered private property.
- **G.** granted, because the police should have secured a warrant before seizing the envelope.
- **H.** denied, because the counselor had a reasonable suspicion of illegal activity.
- **I.** denied, because the discovery of the substance was incidental to the counselor's actions.
- **J.** denied, because the counselor, as a school employee, was in loco parentis.

**Answer: E**

*Source: mmlu_pro_law | Category: humanities | ID: humanities_028*

---

## Q149. A farmer owned a 40-acre tract of farmland located in a small southern town. The farmer leased the property and building thereon to a tenant for a term of seven years commencing on February 15, 2000 and terminating at 12:00 noon on February 15, 2007. The lease contained the following provision:"Lessee covenants to pay the rent of $5,000 per month on the 15th day of each month and to keep the building situated upon said leased premises in as good repair as it was at the time of said lease until the expiration thereof. " The lease also contained a provision giving the tenant the option to purchase 10 acres of the tract for $150,000 at the expiration of the lease term. Before the lease was executed, the farmer orally promised the tenant that he (the farmer) would have the 10-acre tract surveyed. During the last year of the lease, the tenant decided to exercise the option to purchase the 10 acres of the tract. Without the farmer's knowledge, the tenant began to build an irrigation ditch across the northern section of the property. When the tenant notified the farmer that he planned to exercise the option, the farmer refused to perform. The farmer also informed the tenant that he never had the 10-acre tract surveyed. If the tenant brings suit for specific performance, which of the following is the farmer's best defense?

- **A.** The option was unenforceable because it was not included in the written lease.
- **B.** The option agreement was unenforceable under the parol evidence rule.
- **C.** The option to purchase was not exercised within the term of the lease.
- **D.** The tenant failed to pay the full amount of rent as required by the lease.
- **E.** The farmer's promise to survey the tract was an unfulfilled condition precedent to the tenant's right to purchase.
- **F.** The farmer never consented to the tenant's exercise of the option.
- **G.** The tenant's construction of an irrigation ditch constituted a material breach of the lease.
- **H.** The description of the property was too indefinite to permit the remedy sought.
- **I.** The farmer's failure to survey the 10-acre tract excused him from further obligations under the contract.
- **J.** The option was unenforceable because it lacked separate consideration.

**Answer: H**

*Source: mmlu_pro_law | Category: humanities | ID: humanities_029*

---

## Q150. A homeowner died in 1985. His will devised his estate in a southern state to his uncle and his best friend "to share and share alike as tenants in common. "At the time of the homeowner's death, the uncle lived in a different part of the southern state (in which the estate was located), while the best friend resided in a northern state. After the homeowner's funeral, the uncle returned to his own residence, but the best friend decided to occupy the estate. He put his name on the mailbox and has paid the taxes and maintenance expenses. To earn extra money, the best friend rented a small house on the property to a teacher and received a monthly rental payment from her. The best friend also grew fruits on the property and sold them at a stand on Fridays. The uncle has been generally aware of this, but because he cared little about the estate, the uncle has never pressed the best friend about the property. Since 1985 the uncle has not paid any rent or other compensation to the best friend, nor has the best friend requested such payment. In January 2010, a series of disputes arose between the uncle and the best friend for the first time concerning their respective rights to the estate. The state in which the property is located recognizes the usual common law types of cotenancies and follows majority rules on rents and profits. There is no applicable legislation on the subject. The uncle brings an appropriate action for a portion of the proceeds that the best friend received from his fruit stand and a portion of the rent that the teacher paid. If the best friend contests the apportionment of the monies he received, judgment should be for whom?

- **A.** The uncle is entitled to a share of the profits from the best friend's crops, but not the rent paid by the teacher.
- **B.** The best friend is entitled to all profits and rents due to his continuous possession and maintenance of the estate.
- **C.** The uncle is entitled to no share of any of the monies raised because the uncle's lack of contact with the best friend will be deemed a waiver.
- **D.** The uncle has forfeited all claims to the property and its profits due to his lack of interest and engagement.
- **E.** The best friend is entitled to all monies raised due to his investment in the property, regardless of the uncle's claim.
- **F.** The uncle is entitled to all profits and rents due to his blood relation to the deceased homeowner.
- **G.** As a cotenant in possession, the best friend retains the profits from his crops, and the uncle is entitled to a share of the rent paid by the teacher.
- **H.** As a cotenant in possession, the best friend retains the profits from his crops and the rents paid by the teacher.
- **I.** The uncle is entitled to a share of the rent that the teacher paid and the profits from the best friend's crops.
- **J.** The uncle and best friend must evenly split all profits and rents, regardless of who has been maintaining the property.

**Answer: G**

*Source: mmlu_pro_law | Category: humanities | ID: humanities_030*

---

## Q151. A farm and an orchard are adjoining tracts of land located in a county. In 2006, a farmer purchased the farm, a 10-acre tract, in fee simple absolute. The orchard, a 20-acre tract situated to the north of the farm, was owned by a rancher in fee simple absolute. A remote predecessor of the farmer had granted to a shepherd a way for egress and ingress across the farm under such terms and circumstances that an easement appurtenant to the orchard was created. This right-of-way was executed by deed and properly recorded. The shepherd, however, never made any actual use of the right-of-way. In 2010, the rancher conveyed the orchard to the farmer. The next year, the farmer conveyed the orchard by deed to an investor for a consideration of $250,000, receipt of which was acknowledged. Neither the rancher farmer deed nor the farmer  investor deed contained any reference to the easement for right-of-way. The investor has now claimed that she has a right-of-way across the farm. The farmer, on the other hand, has informed the investor that no such easement exists. Assume that both the farm and the orchard abut a public highway and that ingress and egress are afforded the investor by that highway. In an appropriate action by the investor to determine her right to use the right-of-way across the farm, she should

- **A.** win, because the farmer had constructive notice of the easement.
- **B.** lose, because the right-of-way was abandoned inasmuch as there never was any actual use made.
- **C.** lose, because the easement was extinguished by merger when the farmer acquired the orchard from the rancher.
- **D.** lose, because the easement was not in use at the time of the sale to the investor.
- **E.** lose, because the investor has reasonable access to the public highway without using the right-of-way.
- **F.** win, because the investor has a right to access all parts of her property.
- **G.** lose, because the easement was not specifically mentioned in the deed between the farmer and the investor.
- **H.** win, because the right-of-way was never officially terminated.
- **I.** win, because the investor acquired an easement by implication.
- **J.** win, because the original deed clearly states the existence of the right-of-way.

**Answer: C**

*Source: mmlu_pro_law | Category: humanities | ID: humanities_031*

---

## Q152. This question refers to the following information.
"But you, my dear Pangloss," said Candide, "how can it be that I behold you again?"
"It is true," said Pangloss, "that you saw me hanged&….A surgeon purchased my body, carried home, and dissected me. He began with making a crucial incision on me from the navel to the clavicula. One could not have been worse hanged than I was. The executioner of the Holy Inquisition was a sub-deacon, and knew how to burn people marvellously well, but he was not accustomed to hanging. The cord was wet and did not slip properly, and besides it was badly tied; in short, I still drew my breath, when the crucial incision made me give such a frightful scream that my surgeon fell flat upon his back&…[At length he] sewed up my wounds; his wife even nursed me. I was upon my legs at the end of fifteen days&….
One day I took it into my head to step into a mosque, where I saw an old Iman and a very pretty young devotee who was saying her paternosters&….She dropped her bouquet; I picked it up, and presented it to her with a profound reverence. I was so long in delivering it that the Iman began to get angry, and seeing that I was a Christian he called out for help. They carried me before the cadi, who ordered me a hundred lashes on the soles of the feet and sent me to the galleys. I was chained to the very same galley and the same bench as the young Baron. On board this galley there were four young men from Marseilles, five Neapolitan priests, and two monks from Corfu, who told us similar adventures happened daily. The Baron maintained that he had suffered greater injustice than I&….We were continually disputing, and received twenty lashes with a bull's pizzle when the concatenation of universal events brought you to our galley, and you were good enough to ransom us."
"Well, my dear Pangloss," said Candide to him, "when you had been hanged, dissected, whipped, and were tugging at the oar, did you always think that everything happens for the best?"
"I am still of my first opinion," answered Pangloss, "for I am a philosopher and I cannot retract, especially as Leibnitz could never be wrong; and besides, the pre-established harmony is the finest thing in the world, and so is his plenum and materia subtilis."
Voltaire, French Enlightenment writer, Candide, 1759
Candide's statement that "everything always happens for the best" can be seen as a reflection of the Enlightenment belief that

- **A.** a people without a strong central authority are doomed to live in a state of nature
- **B.** humans are inherently corrupt and need strict laws to maintain order
- **C.** only free markets can lead nations to wealth and happiness
- **D.** the only purpose of a government is to secure the rights of life, liberty, and property
- **E.** individuals are the best judges of their own interests
- **F.** only through suffering can one achieve enlightenment
- **G.** society can be perfected if you apply the scientific method to it
- **H.** the world is inherently chaotic and unpredictable
- **I.** religious institutions are the only source of moral authority
- **J.** the universe is a pre-determined and unchangeable system

**Answer: G**

*Source: mmlu_pro_history | Category: humanities | ID: humanities_032*

---

## Q153. This question refers to the following information.
Albeit the king's Majesty justly and rightfully is and ought to be the supreme head of the Church of England, and so is recognized by the clergy of this realm in their convocations, yet nevertheless, for corroboration and confirmation thereof, and for increase of virtue in Christ's religion within this realm of England, and to repress and extirpate all errors, heresies, and other enormities and abuses heretofore used in the same, be it enacted, by authority of this present Parliament, that the king, our sovereign lord, his heirs and successors, kings of this realm, shall be taken, accepted, and reputed the only supreme head in earth of the Church of England, called Anglicans Ecclesia; and shall have and enjoy, annexed and united to the imperial crown of this realm, as well the title and style thereof, as all honors, dignities, preeminences, jurisdictions, privileges, authorities, immunities, profits, and commodities to the said dignity of the supreme head of the same Church belonging and appertaining; and that our said sovereign lord, his heirs and successors, kings of this realm, shall have full power and authority from time to time to visit, repress, redress, record, order, correct, restrain, and amend all such errors, heresies, abuses, offenses, contempts, and enormities, whatsoever they be, which by any manner of spiritual authority or jurisdiction ought or may lawfully be reformed, repressed, ordered, redressed, corrected, restrained, or amended, most to the pleasure of Almighty God, the increase of virtue in Christ's religion, and for the conservation of the peace, unity, and tranquility of this realm; any usage, foreign land, foreign authority, prescription, or any other thing or things to the contrary hereof notwithstanding.
English Parliament, Act of Supremacy, 1534
From the passage and its historical context, one may infer that the Act was, in part,

- **A.** a move to consolidate all religious power under the monarchy
- **B.** a measure to strengthen England's ties with the Catholic Church
- **C.** a response to the threat of invasion by Spain
- **D.** an attempt to legitimize Henry VIII's only heir
- **E.** a move to prevent religious conflict within England
- **F.** an attempt to ally England with the Holy Roman Emperor
- **G.** an attempt to discredit the Pope's authority in England
- **H.** an attempt to prevent the spread of Protestantism in England
- **I.** a solution to Henry VIII's financial difficulties
- **J.** an attempt to establish a state religion in England

**Answer: I**

*Source: mmlu_pro_history | Category: humanities | ID: humanities_033*

---

## Q154. A man hired a videographer to film his daughter's wedding. The written contract entered included a "payment clause," which provided that the videographer would be "paid $10,000 for the filming and editing of a 60-minute video of the wedding and the reception. " The man included in the contract a stipulation that the video would be filmed using high definition equipment. The contract made no other reference to compensation. Thereafter, the videographer filmed and edited the 60-minute video, using high definition equipment, and presented it to the man. The videographer then submitted to the man an invoice statement in the amount of $15,000. Besides the $10,000 contract figure, the bill included a $5,000 charge for the use of the high definition equipment. Denying any additional liability, the man sent the videographer a check for $10,000. The videographer then brought suit against the man to recover the additional $5,000. Which of the following arguments would be most persuasive to support the videographer's contention that when the written contract was executed, the man agreed to pay the videographer $5,000 for use of the high definition equipment in addition to the $10,000 fee?

- **A.** The contract is open to interpretation and does not explicitly state that use of high definition equipment would be included in the $10,000 fee.
- **B.** The use of high definition equipment is a separate service and not included in the base fee for filming and editing.
- **C.** According to the customary trade practice of the video industry, a $10,000 fee for filming and editing means $10,000 in addition to a supplemental charge if high definition equipment is used.
- **D.** An oral agreement to that effect, if provable, would only supplement, not contradict, the "payment clause" as written.
- **E.** The man's stipulation for high definition equipment implies agreement to additional charges associated with the use of such equipment.
- **F.** The videographer can provide evidence of past clients who were charged separately for the use of high definition equipment.
- **G.** Assuming arguendo that the written "payment clause" was fully integrated and neither patently nor latently ambiguous, equitable considerations require admission of extrinsic evidence, if available, of the parties' intent, since the videographer would stand to lose $5,000 on the contract.
- **H.** Under the UCC, extrinsic evidence, if available, of additional terms agreed to by the parties is admissible unless such terms "would certainly vary or contradict those contained in the document. "
- **I.** The videographer had previously informed the man of the additional costs of using high definition equipment.
- **J.** The videographer provided a service above and beyond the agreed upon terms, therefore justifying the additional cost.

**Answer: C**

*Source: mmlu_pro_law | Category: humanities | ID: humanities_034*

---

## Q155. This question refers to the following information.
"The law of love, peace and liberty in the states extending to Jews, Turks and Egyptians, as they are considered sonnes of Adam, which is the glory of the outward state of Holland, soe love, peace and liberty, extending to all in Christ Jesus, condemns hatred, war and bondage. And because our Saviour sayeth it is impossible but that offences will come, but woe unto him by whom they cometh, our desire is not to offend one of his little ones, in whatsoever form, name or title hee appears in, whether Presbyterian, Independent, Baptist or Quaker, but shall be glad to see anything of God in any of them, desiring to doe unto all men as we desire all men should doe unto us, which is the true law both of Church and State; for our Saviour sayeth this is the law and the prophets.
"Therefore if any of these said persons come in love unto us, we cannot in conscience lay violent hands upon them, but give them free egresse and regresse unto our Town, and houses, as God shall persuade our consciences, for we are bounde by the law of God and man to doe good unto all men and evil to noe man. And this is according to the patent and charter of our Towne, given unto us in the name of the States General, which we are not willing to infringe, and violate, but shall houlde to our patent and shall remaine, your humble subjects, the inhabitants of Vlishing (Flushing, part of the colony of New Netherlands)."
—The Flushing Remonstrance, 1657
Which of the following was most significant in enshrining into the U.S. legal structure the ideas contained in the Flushing Remonstrance?

- **A.** The enumeration of congressional powers in the Constitution.
- **B.** The "Right to Bear Arms" clause of the Second Amendment.
- **C.** The 14th Amendment's "Equal Protection Clause"
- **D.** The "Due Process Clause" of the Fifth Amendment.
- **E.** The "Right to a Speedy and Public Trial" clause of the Sixth Amendment.
- **F.** The "Cruel and Unusual Punishment" clause of the Eighth Amendment.
- **G.** The "Establishment Clause" of the First Amendment.
- **H.** The "Free Exercise Clause" of the First Amendment.
- **I.** The preamble of the Declaration of Independence.
- **J.** The "Double Jeopardy" clause of the Fifth Amendment.

**Answer: H**

*Source: mmlu_pro_history | Category: humanities | ID: humanities_035*

---

## Q156. Proposed legislation was offered to a state legislature that would reorganize the state police. The bill created a great deal of controversy, both in and outside the state government. Several leaders of the minority party in the legislature decided to oppose the legislation. One member of the minority party disagreed with his party's opposition to the bill and publicly announced his support for the legislation. The minority party leaders called a caucus to discuss and determine their legislative strategy for floor debate on the bill. When the disagreeing member appeared at the door of the caucus room, he was denied admission because of his anti-party stance. He was also informed that he would be removed from all of his committee assignments. During the caucus, the party members discussed other means of disciplining the member for his party insubordination. It was suggested that they issue a press release in which the party would publicly castigate him for his actions. The leader of the party said that "the member is a cutthroat politician who is only looking out for where his next buck will come from. "Which of the following constitutional provisions would give the ousted member his best grounds for challenging his exclusion from the party caucus?

- **A.** The speech and debate clause.
- **B.** The establishment clause of the First Amendment.
- **C.** The due process clause of the Fourteenth Amendment.
- **D.** The right to petition as guaranteed by the First Amendment.
- **E.** The right to a jury trial as guaranteed by the Sixth Amendment.
- **F.** The right of assembly as guaranteed by the First Amendment.
- **G.** The equal protection clause of the Fourteenth Amendment.
- **H.** The protection from ex post facto laws.
- **I.** The cruel and unusual punishment clause of the Eighth Amendment.
- **J.** The privileges and immunities clause of the Fourteenth Amendment.

**Answer: C**

*Source: mmlu_pro_law | Category: humanities | ID: humanities_036*

---

## Q157. This question refers to the following information.
Perhaps, however, I am more conscious of the importance of civil liberties in this particular moment of our history than anyone else, because I travel through the country and meet people and see things that have happened to little people, I realize what it means to democracy to preserve our civil liberties.
All through the years we have had to fight for civil liberty, and we know that there are times when the light grows rather dim, and every time that happens democracy is in danger. Now, largely because of the troubled state of the world as a whole, civil liberties have disappeared in many other countries.
It is impossible, of course, to be at war and keep freedom of the press and freedom of speech and freedom of assembly. They disappear automatically. And so in many countries where ordinarily they were safe, today they have gone. In other countries, even before war came, not only freedom of the press and freedom of assembly, and freedom of speech disappeared, but freedom of religion disappeared.
And so we know here in this country, we have a grave responsibility. We are at peace. We have no reason for the fears which govern so many other peoples throughout the world; therefore, we have to guard the freedoms of democracy.
—Eleanor Roosevelt, Address to the American Civil Liberties Union, Chicago, Illinois, March 14, 1940
Roosevelt's concerns can most directly be compared to those of the people who debated which of the following?

- **A.** The Homestead Act of 1862
- **B.** The USA Patriot Act of 2001
- **C.** The Gulf of Tonkin Resolution of 1964
- **D.** The Affordable Care Act of 2010
- **E.** The Civil Rights Act of 1964
- **F.** The Indian Removal Act of 1830
- **G.** The Social Security Act of 1935
- **H.** The Defense of Marriage Act of 1996
- **I.** The Voting Rights Act of 1965
- **J.** The Alien and Sedition Acts of 1798

**Answer: B**

*Source: mmlu_pro_history | Category: humanities | ID: humanities_037*

---

## Q158. This question refers to the following information.
BECAUSE no People can be truly happy, though under the greatest Enjoyment of Civil Liberties, if abridged of the Freedom of their Consciences, as to their Religious Profession and Worship: And Almighty God being the only Lord of Conscience, Father of Lights and Spirits; and the Author as well as Object of all divine Knowledge, Faith and Worship, who only doth enlighten the Minds, and persuade and convince the Understanding of People, I do hereby grant and declare, That no Person or Persons, inhabiting in this Province or Territories, who shall confess and acknowledge One almighty God, the Creator, Upholder and Ruler of the World; and profess him or themselves obliged to live quietly under the Civil Government, shall be in any Case molested or prejudiced, in his or their Person or Estate, because of his or their conscientious Persuasion or Practice, nor be compelled to frequent or maintain any religious Worship, Place or Ministry, contrary to his or their Mind.…
—William Penn, Charter of Privileges Granted by William Penn,
esq. to the Inhabitants of Pennsylvania and Territories, October 28, 1701
Because of Penn's Charter of Privileges, Pennsylvania became

- **A.** a colony that banned all forms of religious expression.
- **B.** one of the least religiously diverse colonies in America.
- **C.** one of the most religiously diverse colonies in British America.
- **D.** known for its strict adherence to a single religious doctrine.
- **E.** known for its hostility to traditional religious practices.
- **F.** a colony that forbade the practice of any religion other than Christianity.
- **G.** a colony known for religious repression and intolerance.
- **H.** notorious for witch hunting and popular superstition.
- **I.** the only colony in America where religion was not practiced.
- **J.** a colony where religious diversity was punishable by law.

**Answer: C**

*Source: mmlu_pro_history | Category: humanities | ID: humanities_038*

---

## Q159. In 1888, a landowner owned a dairy farm. The landowner conveyed this real property to his son in1938. In 1953, the son conveyed the dairy farm to his friend. This deed was not recorded until after the son's death in 1957. In 1956, the son mortgaged the dairy farm to the bank. The mortgage instrument, which was recorded in 1956, recited that it was subordinate to a mortgage on the same land given by the son to an investor in 1936 and recorded in 1936. In that instrument the son purported to grant the investor a mortgage on the dairy farm. In 1979, the friend conveyed the dairy farm to a farmer. This deed was duly recorded, but did not mention any mortgage. In 2008, a buyer entered into an agreement with the farmer, whereby the farmer would convey the dairy farm in fee simple to the buyer for the sum of $75,000. The closing date was set for January 15, 2009. All of the deeds mentioned in the aforementioned transactions are general warranty deeds. In addition, this jurisdiction has a notice-type recording statute and follows a title theory for mortgages. On January 15, 2009, the sale of the dairy farm is finalized and the buyer paid the farmer $75,000. The fanner executed a general warranty deed. The deed contains the following covenants of title:(1) Covenant for seisin. (2) Covenant of the right to convey. (3) Covenant against encumbrances. After the buyer takes possession of the dairy farm, he learns of the son investor 1936 mortgage, which was not satisfied, and seeks monetary damages for breach of the covenant against encumbrances. Judgment should be for

- **A.** the farmer, unless the covenantee is disturbed in his actual enjoyment of the land thereby conveyed.
- **B.** the buyer, if the farmer knew about the mortgage to the investor but did not disclose it.
- **C.** the buyer, because the covenant against encumbrances protects against future claims on the property.
- **D.** the buyer, because the covenant of the right to convey was breached.
- **E.** the farmer, because the son's mortgage to the investor was not mentioned in the deed.
- **F.** the buyer, because the covenant against encumbrances is a guarantee to the grantee that the property is not subject to outstanding rights or interests.
- **G.** the farmer, if the mortgage to the investor was satisfied before the sale to the buyer.
- **H.** the buyer, because the covenant against encumbrances would be breached at the time the deed was delivered, thereby entitling the covenantee to recover damages.
- **I.** the farmer, because the buyer did not perform a thorough title search before purchasing the property.
- **J.** the farmer, because the covenant against encumbrances may only be breached, if at all, at the time of conveyance.

**Answer: H**

*Source: mmlu_pro_law | Category: humanities | ID: humanities_039*

---

## Q160. This question refers to the following information.
"Wherever I go—the street, the shop, the house, or the steamboat—I hear the people talk in such a way as to indicate that they are yet unable to conceive of the Negro as possessing any rights at all. Men who are honorable in their dealings with their white neighbors will cheat a Negro without feeling a single twinge of their honor. To kill a Negro they do not deem murder; to debauch a Negro woman they do not think fornication; to take the property away from a Negro they do not consider robbery. The people boast that when they get freedmen affairs in their own hands, to use their own classic expression, 'the niggers will catch hell.'
"The reason of all this is simple and manifest. The whites esteem the blacks their property by natural right, and however much they may admit that the individual relations of masters and slaves have been destroyed by the war and the President's emancipation proclamation, they still have an ingrained feeling that the blacks at large belong to the whites at large, and whenever opportunity serves they treat the colored people just as their profit, caprice or passion may dictate."
—Congressional testimony of Col. Samuel Thomas, Assistant Commissioner, Bureau of Refugees, Freedmen and Abandoned Lands, 1865
Which of the following specific developments contributed to the general sentiment expressed in Federalist #15?

- **A.** Dutch traders forced American ships to extend tribute payments in order to dock in Holland.
- **B.** Russian explorers claimed land in the Pacific Northwest.
- **C.** Portugal imposed trade restrictions on American goods.
- **D.** Spain denied American access to the Mississippi River.
- **E.** Italian mercenaries attempted to seize control of the Mississippi River.
- **F.** Great Britain refused to evacuate forts in the Great Lakes region.
- **G.** French and Spanish forces engaged in skirmishes with American settlers on the western frontier.
- **H.** British forces supported American Indian tribes in conducting raids on the Frontier.
- **I.** French forces aided American Indians in conducting raids on New England.
- **J.** Spanish forces retook Florida from the United States.

**Answer: F**

*Source: mmlu_pro_history | Category: humanities | ID: humanities_040*

---

## Q161. A firm in a perfectly competitive industry has patented a new process for making widgets. The new process lowers the firm's average cost, meaning that this firm alone (although still a price taker) can earn real economic profits in the long run. Suppose a government study has found that the firm's new process is polluting the air and estimates the social marginal cost of widget production by this firm to be SMC = 0.5q. If the market price is $20, what should be the rate of a government-imposed excise tax to bring about optimal level of production?

- **A.** 14
- **B.** 18
- **C.** 8
- **D.** 2
- **E.** 4
- **F.** 20
- **G.** 10
- **H.** 12
- **I.** 16
- **J.** 6

**Answer: E**

*Source: mmlu_pro_business | Category: general | ID: general_001*

---

## Q162. The McDougall Company produces and sells two qualities of carpet. The better quality carpeting sells for $6.00 per yard and has variable costs of 70 percent of sales. The other type costs $3.00 per yard and has variable costs of 60 percent of sales. Total fixed costs of both types of carpet combined are $122,200. If both carpets have the same dollar sales, how many yards of each must be sold to earn a profit of $50,000?

- **A.** 30000 yards and 90000 yards
- **B.** 48500 yards and 72750 yards
- **C.** 45000 yards and 75000 yards
- **D.** 47833(1/3) yards and 71750 yards
- **E.** 50000 yards and 60000 yards
- **F.** 52000 yards and 78000 yards
- **G.** 55000 yards and 65000 yards
- **H.** 35000 yards and 70000 yards
- **I.** 40000 yards and 80000 yards
- **J.** 60000 yards of each type

**Answer: D**

*Source: mmlu_pro_business | Category: general | ID: general_002*

---

## Q163. The Experimental Company is now producing and selling 17,000 units of product per year at a sales price of $250 each. It has had a profit objective of 20 percent return on a $2,000,000 investment. Its present cost structure is as follows: Manufacturing costs: Fixed costs $400,000 per year Variable costs $175 per unit produced Selling and administration expenses: Fixed expenses $135,000 per year Variable expenses $25 per unit sold Ten hours are required to produce a unit of product. The company is interested in pushing its rate of operations to 100 percent of its capacity-20,000 units of product. It believes that it can raise its price by up to 5 percent of the present sales price without cutting its present sales volume. In addition, it believes that it can increase its sales to productive capacity with minor design and quality improvements that will increase variable manufacturing costs by $5 per unit, and with accompanying promotion that will increase its variable selling and administration expenses by $5 per unit for all units sold. The company would like to achieve a profit objective of 25 percent on its investment. REQUIRED: 1) Can the company, within the price increase limitation indicated, achieve its new profit objective by reaching full utilization of capacity with the increased variable costs and expenses required? 2) What would the sales price have to be to achieve the new profit objective? 3) What sales price would achieve the old profit objective?

- **A.** $258.00, $259.50/unit, $253.00/unit
- **B.** $257.50, $258.75/unit, $252.00/unit
- **C.** $269.00, $269.50/unit, $263.00/unit
- **D.** $262.50, $261.75/unit, $255.00/unit
- **E.** $255.50, $262.75/unit, $265.00/unit
- **F.** $267.50, $268.00/unit, $262.50/unit
- **G.** $270.00, $270.50/unit, $265.00/unit
- **H.** $263.50, $264.00/unit, $258.50/unit
- **I.** $265.50, $265.75/unit, $260.00/unit
- **J.** $260.50, $260.75/unit, $250.00/unit

**Answer: D**

*Source: mmlu_pro_business | Category: general | ID: general_003*

---

## Q164. A company wants a 20 percent return on investment before taxes at a sales volume equal to 80 percent of capacity. Fixed annual costs are $200,000 and the annual capacity is 200,00 units. If the variable cost per unit is $9 and the company investment is $1,400,000, what should be the selling price per unit given that production and sales are 80 percent of capacity?

- **A.** $15
- **B.** $18
- **C.** $13
- **D.** $17
- **E.** $19
- **F.** $11
- **G.** $12
- **H.** $10
- **I.** $16
- **J.** $14

**Answer: G**

*Source: mmlu_pro_business | Category: general | ID: general_004*

---

## Q165. In corporation XY sales in year 197X amount to $3,000,000; the cost of goods sold is $1,500,000; administrative ex-penses are $300,000; depreciation $800,000; all other ex-penses are $100,000; retained earnings are $100,000; when the corporation income tax rate is 48%, how much dividend is declared? How much could be declared if the tax rate fell to 45%?

- **A.** $58,000 and $68,000
- **B.** $55,000 and $66,000
- **C.** $56,000 and $65,000
- **D.** $52,000 and $62,000
- **E.** $60,000 and $70,000
- **F.** $59,000 and $69,000
- **G.** $53,000 and $63,000
- **H.** $50,000 and $60,000
- **I.** $54,000 and $64,000
- **J.** $57,000 and $67,000

**Answer: C**

*Source: mmlu_pro_economics | Category: general | ID: general_005*

---

## Q166. Which of the following statements are true concerning a triangular or recursive system?

i) The parameters can be validly estimated using separate applications of OLS to

each equation


ii) The independent variables may be correlated with the error terms in other

equations


iii) An application of 2SLS would lead to unbiased but inefficient parameter estimates


iv) The independent variables may be correlated with the error terms in the equations

in which they appear as independent variables

- **A.** (i), (ii), (iii), and (iv)
- **B.** (iii) and (iv) only
- **C.** (i), (ii), and (iii) only
- **D.** (i) and (iii) only
- **E.** (i), (iii), and (iv) only
- **F.** (ii) only
- **G.** (ii) and (iii) only
- **H.** (i) and (iv) only
- **I.** (i) and (ii) only
- **J.** (ii) and (iv) only

**Answer: C**

*Source: mmlu_pro_economics | Category: general | ID: general_006*

---

## Q167. An accountant has been engaged to compile a nonissuer's financial statements that contain several misapplications of accounting principles and unreasonable accounting estimates. Management is unwilling to revise the financial statements and the accountant believes that modification of the standard compilation report is not adequate to communicate the deficiencies. Under these circumstances the accountant should

- **A.** Issue a standard compilation report without any modifications.
- **B.** Withdraw from the compilation engagement and provide no further services concerning these financial statements.
- **C.** Advise the management to hire another accountant for the engagement.
- **D.** Inform the regulatory authorities about the misapplications of accounting principles.
- **E.** Inform management that the engagement can proceed only if distribution of the accountant's compilation report is restricted to internal use.
- **F.** Continue the engagement and issue a modified report without informing the board of directors.
- **G.** Disclaim an opinion on the financial statements and advise the board of directors that the financial statements should not be relied upon.
- **H.** Proceed with the engagement but add a disclaimer in the report that the accountant is not responsible for the misapplications.
- **I.** Proceed with the engagement but refuse to sign the compilation report.
- **J.** Determine the effects of the deficiencies and add a separate paragraph to the compilation report that describes the deficiencies and their effects.

**Answer: B**

*Source: mmlu_pro_other | Category: general | ID: general_007*

---

## Q168. At the Harris foundry, the total daily cost of manufacturing m pieces of metal is represented by the following equation: C (in dollars) = 3 m^2 + m + 9. If the cost of the work produced by 9:00 am is $4,849 and the rate of manufacture after 9:00 am is 10 pieces per hr., (i) how many pieces of metal will have been produced by lunch hour (12 noon)? Hint: use the quadratic formula: m = [{- b \pm \surd(b^2 - 4ac)} / {2a}], and other calculations. (ii) When t represents the number of hours past 9:00 am., express the total cost as a function of t.

- **A.** 90 pieces
- **B.** 50 pieces
- **C.** 60 pieces
- **D.** 100 pieces
- **E.** 95 pieces
- **F.** 70 pieces
- **G.** 85 pieces
- **H.** 65 pieces
- **I.** 75 pieces
- **J.** 80 pieces

**Answer: F**

*Source: mmlu_pro_business | Category: general | ID: general_008*

---

## Q169. Mr. Golden purchased 3 bonds, each with a maturity value of $1,000, from theSuttonsmithCorporation. For each bond, he will receive $15 semiannually for 20 years, after which time he will also receive the full face value of $1,000. The $15 payments will be made regardless of the interest rate. If the interest rate on one bond was 3%; on another, 4%; and on the third, 3.6%, what did Mr. Golden pay for each bond?

- **A.** $950.00, $800.23, $850.26
- **B.** $1,000.04, $863.23, $915.26
- **C.** $1,000.04, $860.23, $910.26
- **D.** $1,050.00, $890.23, $940.26
- **E.** $1,000.00, $860.23, $910.26
- **F.** $1,050.04, $813.23, $935.26
- **G.** $995.00, $855.23, $905.26
- **H.** $1,000.04, $865.23, $915.26
- **I.** $1,000.00, $900.23, $950.26
- **J.** $1,000.04, $863.23, $920.26

**Answer: B**

*Source: mmlu_pro_business | Category: general | ID: general_009*

---

## Q170. Two power line construction routes are being considered. Route A is 15 miles long and goes around a lake. Each mile will cost $6,000 to build and $2,000 a year to maintain. At the end of fifteen years, each mile will have a salvage value of $3,000. Route B is an underwater line that cuts 5 miles across the lake. Construction costs will be $31,000 per mile and annual maintenance costs, $400 per mile. The salvage value at the end of fifteen years will be $6,000 per mile. Assuming interest is 8% and taxes are 3% of the construction costs of each power line, compare the annual costs of Route A and Route B for the first year.

- **A.** $41,558 for Route A and $23,654 for Route B
- **B.** $45,000 for Route A and $30,000 for Route B
- **C.** $100,000 for Route A and $50,000 for Route B
- **D.** $40,000 for Route A and $20,000 for Route B
- **E.** $80,000 for Route A and $40,000 for Route B
- **F.** $35,000 for Route A and $25,000 for Route B
- **G.** $70,000 for Route A and $35,000 for Route B
- **H.** $60,000 for Route A and $155,000 for Route B
- **I.** $90,000 for Route A and $155,000 for Route B
- **J.** $120,000 for Route A and $75,000 for Route B

**Answer: A**

*Source: mmlu_pro_business | Category: general | ID: general_010*

---

## Q171. Assume the following model (from the preceding problem). Y = C + I + G C = 100 + 0.6Y I = 0.2Y - 50i M_D = 0.25Y - 30i M_s = 65 G = 100 whose equilibrium level was found to be 500. Suppose that full employment level of income is 600, so that the desired change is 100. If the money supply is held constant, what change in govern-ment spending will be required to close the deflationary gap?

- **A.** 80
- **B.** 75
- **C.** 55
- **D.** 90
- **E.** 50
- **F.** 61.5
- **G.** 70
- **H.** 85
- **I.** 100
- **J.** 65

**Answer: F**

*Source: mmlu_pro_business | Category: general | ID: general_011*

---

## Q172. Mr. Louis is presently considering buying a new boat to give rides to tourists. He has two alternatives: Boat A costs $10,000 and consumes $2,000 in fuel per year. Boat B costs $7,000 and consumes $2,500. Both boats have a zero salvage value at the end of 10 years. If Ur. Louis considers a rate of return of 6% acceptable, (a) which boat should he purchase? (b) how much will he charge each tourist if there are 3 tourists to a ride and Mr. Louis plans to have 125 rides each year?

- **A.** Boat B, $9.50 per passenger
- **B.** Boat A, $8.96 per passenger
- **C.** Boat A, $9.50 per passenger
- **D.** Boat A, $12 per passenger
- **E.** Boat A, $7.50 per passenger
- **F.** Boat B, $7.50 per passenger
- **G.** Boat B, $8.96 per passenger
- **H.** Boat B, $10 per passenger
- **I.** Boat B, $12 per passenger
- **J.** Boat A, $10 per passenger

**Answer: B**

*Source: mmlu_pro_business | Category: general | ID: general_012*

---

## Q173. Consider a forward contract on a 4-year bond with maturity 1 year. The current value of the bond is $1018.86, it has a face value of $1000 and a coupon rate of 10% per annum. A coupon has just been paid on the bond and further coupons will be paid after 6 months and after 1 year, just prior to delivery. Interest rates for 1 year out are flat at 8%. Compute the forward price of the bond.

- **A.** 960.40
- **B.** 1015.30
- **C.** 1030.88
- **D.** 990.90
- **E.** 999.998976
- **F.** 1050.75
- **G.** 1020.50
- **H.** 980.65
- **I.** 975.20
- **J.** 1001.10

**Answer: E**

*Source: mmlu_pro_business | Category: general | ID: general_013*

---

## Q174. A shoe retailer allows customers to return shoes within 90 days of purchase. The company estimates that 5% of sales will be returned within the 90-day period. During the month, the company has sales of $200,000 and returns of sales made in prior months of $5,000. What amount should the company record as net sales revenue for new sales made during the month?

- **A.** $195,000
- **B.** $200,000
- **C.** $205,000
- **D.** $220,000
- **E.** $175,000
- **F.** $180,000
- **G.** $185,000
- **H.** $210,000
- **I.** $215,000
- **J.** $190,000

**Answer: J**

*Source: mmlu_pro_other | Category: general | ID: general_014*

---

## Q175. World Bank data show that in 1995, the poorest 20% of households accounted for 7.5% of household income in Niger, the next 20% of households accounted for 11.8% of income, the middle 20% accounted for 15.5% of income, the second richest 20% accounted for 21.1% of income, and the top 20% accounted for 44.1% of income. What was the cumulative income share of the bottom 60% of households in Niger?

- **A.** 48.10%
- **B.** 34.80%
- **C.** 44.10%
- **D.** 56.40%
- **E.** 7.50%
- **F.** 29.60%
- **G.** 21.10%
- **H.** 11.80%
- **I.** 15.50%
- **J.** 65.20%

**Answer: B**

*Source: mmlu_pro_other | Category: general | ID: general_015*

---

## Q176. At the beginning of the 19X1 fiscal year Company X had $28,000 of accounts receivable. At the end of the fiscal year it had $32,000. of accounts receivable. Sales in 19X1 were $850,000. At the end of the 19X2 fiscal year, accounts receivable were $35,000. Sales in 19X2 were $920,000. Using a 360 day year and given that Company X's desired rate of return is 10%, (a) find the average collection period (in days) for 19X1 and 19X2 (b) find the cost (or saving) in extending (or reducing) the credit period during the two periods.

- **A.** 13.52 days, 14.23 days, $5.67
- **B.** 10.5 days, 11.4 days, $3.50
- **C.** 11.2 days, 12.1 days, $3.75
- **D.** 11.9 days, 12.8 days, $3.98
- **E.** 12.2 days, 12.9 days, $4.15
- **F.** 14.7 days, 15.6 days, $6.20
- **G.** 13.3 days, 14.1 days, $5.10
- **H.** 12.7 days, 13.11 days, $4.45
- **I.** 15.8 days, 16.7 days, $7.26
- **J.** 14.2 days, 15.1 days, $6.78

**Answer: H**

*Source: mmlu_pro_business | Category: general | ID: general_016*

---

## Q177. Adam Smith is considering automating his pin factory with the purchase of a $475,000 machine. Shipping and installation would cost $5,000. Smith has calculated that automation would result in savings of $45,000 a year due to reduced scrap and $65,000 a year due to reduced labor costs. The machine has a useful life of 4 years and falls in the 3-year property class for MACRS depreciation purposes. The estimated final salvage value of the machine is $120,000. The firm's marginal tax rate is 34 percent. The incremental cash outflow at time period 0 is closest to

- **A.** $280,000.00
- **B.** $180,000.00
- **C.** $580,000.00
- **D.** $700,000.00
- **E.** $420,000.00
- **F.** $500,000.00
- **G.** $480,000.00
- **H.** $380,000.00
- **I.** $600,000.00
- **J.** $300,000.00

**Answer: G**

*Source: mmlu_pro_other | Category: general | ID: general_017*

---

## Q178. During 1979, Mr. Anderson expected to earn $20,000. From thisincome he had planned to save $2,000. However, during 1979, Mr. Anderson got a raise which boosted his income to $23,000.If Mr. Anderson ended up saving a total of $3,000 outof his $23,000 income, what was his marginal propensity toconsume (MPC) ? (It may be assumed that if he had not receivedhis raise, Mr. Anderson would have actually saved the$2,000 that he had planned to save.)

- **A.** 3/5
- **B.** 2/3
- **C.** 3/4
- **D.** 4/5
- **E.** 1/4
- **F.** 7/8
- **G.** 1/2
- **H.** 2/5
- **I.** 1/3
- **J.** 5/6

**Answer: B**

*Source: mmlu_pro_economics | Category: general | ID: general_018*

---

## Q179. Traders in major financial institutions use the Black-Scholes formula in a backward fashion to infer other traders' estimation of $\sigma$ from option prices. In fact, traders frequently quote sigmas to each other, rather than prices, to arrange trades. Suppose a call option on a stock that pays no dividend for 6 months has a strike price of $35, a premium of $2.15, and time to maturity of 7 weeks. The current short-term T-bill rate is 7%, and the price of the underlying stock is $36.12. What is the implied volatility of the underlying security?

- **A.** 0.275
- **B.** 0.225
- **C.** 0.165
- **D.** 0.195
- **E.** 0.210
- **F.** 0.350
- **G.** 0.300
- **H.** 0.180
- **I.** 0.320
- **J.** 0.251

**Answer: J**

*Source: mmlu_pro_business | Category: general | ID: general_019*

---

## Q180. Suppose that the value of $R^2$ for an estimated regression model is exactly zero. Which of the following are true?

i) All coefficient estimates on the slopes will be zero

ii) The fitted line will be horizontal with respect to all of the explanatory variables

iii) The regression line has not explained any of the variability of y about its mean value

iv) The intercept coefficient estimate must be zero.

- **A.** (i), (ii), (iii), and (iv)
- **B.** (i) and (ii) only
- **C.** (i) and (iii) only
- **D.** (iii) and (iv) only
- **E.** (ii) and (iii) only
- **F.** (ii) and (iv) only
- **G.** (ii), (iii), and (iv) only
- **H.** (i) and (iv) only
- **I.** (i), (iii), and (iv) only
- **J.** (i), (ii), and (iii) only

**Answer: J**

*Source: mmlu_pro_economics | Category: general | ID: general_020*

---

## Q181. Star Co. is a retail store specializing in contemporary furniture. The following information is taken from Star's June budget: Sales $540000 Cost of goods sold 300000 Merchandise inventory‚ÄìJune 1 150000 Merchandise inventory‚ÄìJune 30 180000 Accounts payable for purchases‚ÄìJune 1 85000 Accounts payable for purchases‚ÄìJune 30 75000 What amount should Star budget for cash disbursements for June purchases?

- **A.** $310,000
- **B.** $320,000
- **C.** $260,000
- **D.** 340000
- **E.** $280,000
- **F.** $380,000
- **G.** $300,000
- **H.** $350,000
- **I.** $330,000
- **J.** $360,000

**Answer: D**

*Source: mmlu_pro_other | Category: general | ID: general_021*

---

## Q182. Three years ago, Fred invested $10,000 in the shares of ABC Corp. Each year, the company distributed dividends to its shareholders. Each year, Fred received $100 in dividends. Note that since Fred received $100 in dividends each year, his total income is $300. Today, Fred sold his shares for $12,000. What is the holding period return of his investment?

- **A.** 0.28
- **B.** 0.15
- **C.** 0.18
- **D.** 0.40
- **E.** 0.25
- **F.** 0.30
- **G.** 0.33
- **H.** 0.10
- **I.** 0.23
- **J.** 0.20

**Answer: I**

*Source: mmlu_pro_business | Category: general | ID: general_022*

---

## Q183. Pine Co. purchased land for $450000 as a factory site. An existing building on the site was razed before construction began. Additional information is as follows: Cost of razing old building $60000 Title insurance and legal fees to purchase land $30000 Architect's fees $95000 New building construction cost $1850000 What amount should Pine capitalize as the cost of the completed factory building?

- **A.** $1,990,000
- **B.** 1910000
- **C.** $1,945,000
- **D.** $1,915,000
- **E.** $2,005,000
- **F.** $1,950,000
- **G.** $1,920,000
- **H.** $1,975,000
- **I.** $2,025,000
- **J.** $1,975,500

**Answer: C**

*Source: mmlu_pro_other | Category: general | ID: general_023*

---

## Q184. Prior to the issuance of its December 31 financial statements Stark Co. was named as a defendant in a lawsuit arising from an event that occurred in October. Stark's legal counsel believes that it is reasonably possible that there will be an unfavorable outcome and that damages will range from $100000 to $150000. Which amount(s) should Stark accrue and/or disclose in its December 31 financial statements? Accrue contingent liability Disclose contingent liability

- **A.** $150000 $0
- **B.** $0 $150000
- **C.** $0 $0
- **D.** $0 $100000 - $150000
- **E.** $100000 $150000
- **F.** $100000 $0
- **G.** $100000 $100000
- **H.** $150000 $150000
- **I.** $150000 $100000 - $150000
- **J.** $100000 $100000 - $150000

**Answer: D**

*Source: mmlu_pro_other | Category: general | ID: general_024*

---

## Q185. The City of Windemere decided to construct several large windmills to generate electrical power. The construction was financed through a general residential property tax levy for the next ten years. Utility revenues are intended to offset all expenses associated with the windmills. The land for the windmills was donated to the city by a local farmer. The land from the farmer should be reported in which fund type?

- **A.** Permanent.
- **B.** Internal Service.
- **C.** Trust and Agency.
- **D.** Capital projects.
- **E.** Enterprise.
- **F.** Debt Service.
- **G.** General Fund.
- **H.** Pension (and other employee benefit) Trust.
- **I.** Investment Trust.
- **J.** Special revenue.

**Answer: E**

*Source: mmlu_pro_other | Category: general | ID: general_025*

---

## Q186. Mr. Fields is considering selling a property acquired 15 years ago for $65,000; $10,000 was for the land, and $55,000 was for the building. Mr. Fields can now sell the property for $120,000. He can also keep it and continue to collect the annual rent from the building. If Mr. Fields decides to keep the property, he will keep it for another 25 years and will, at that time, sell it for $40,000. The annual rent receipts will be $5,930 for the next twenty-five years. Should Mr. Fields sell the property now or in twenty-five years? Assume that the building, but not the land, has been depreciated using the straight line method at 2%, long term gains are taxable at a rate of 25%, Mr. Fields' income tax rate is 56%, the minimum attractive rate of return after taxes is 2(1/2)%, and taxes will hold for the next 25 years. Make all calculations to the nearest dollar.

- **A.** Mr. Fields should lease the property to a new tenant for a higher annual rent
- **B.** Mr. Fields should sell the property now for $120,000
- **C.** Mr. Fields should keep the property for 25 more years
- **D.** Mr. Fields should convert the building into a personal residence
- **E.** Mr. Fields should rent the property for another 15 years
- **F.** Mr. Fields should sell the property after 10 years
- **G.** Mr. Fields should renovate the property and then decide whether to sell or keep it
- **H.** Mr. Fields should exchange the property for a similar one to defer taxes
- **I.** Mr. Fields should hold the property indefinitely for future generations
- **J.** Mr. Fields should donate the property to a charitable organization for a tax write-off

**Answer: B**

*Source: mmlu_pro_business | Category: general | ID: general_026*

---

## Q187. Match Co. manufactures a product with the following costs per unit based on a maximum plant capacity of 400000 units per year: Direct materials $ 60 Direct labor 10 Variable overhead 40 Fixed overhead 30 Total $140 Match has a ready market for all 400000 units at a selling price of $200 each. Selling costs in this market consist of $10 per unit shipping and a fixed licensing fee of $50000 per year. Reno Co. wishes to buy 5000 of these units on a special order. There would be no shipping costs on this special order. What is the lowest price per unit at which Match should be willing to sell the 5000 units to Reno?

- **A.** $160
- **B.** $110
- **C.** $190
- **D.** $180
- **E.** $230
- **F.** 200
- **G.** $240
- **H.** $220
- **I.** $210
- **J.** $140

**Answer: C**

*Source: mmlu_pro_other | Category: general | ID: general_027*

---

## Q188. Boy Alcott and Jon Buxton are partners in a steel company. They share the net income in proportion to their average investments. On January 1, Alcott invested $4,000 and Buxton invested $5,000. On May 1, Alcott invested an additional $2,000 and Buxton invested $1,750. On September 1, Alcott withdrew $500. On November 1, each partner invested an additional $2,000. The net profit for the year was $8,736. Find each partner's share of the profit.

- **A.** Alcott's share: $3,936, Buxton's share: $4,800
- **B.** Alcott's share: $4,004, Buxton's share: $4,732
- **C.** Alcott's share: $4,200, Buxton's share: $4,536
- **D.** Alcott's share: $4,800, Buxton's share: $3,936
- **E.** Alcott's share: $5,000, Buxton's share: $3,736
- **F.** Alcott's share: $3,868, Buxton's share: $4,868
- **G.** Alcott's share: $4,368, Buxton's share: $4,368
- **H.** Alcott's share: $4,732, Buxton's share: $4,004
- **I.** Alcott's share: $4,500, Buxton's share: $4,236
- **J.** Alcott's share: $5,236, Buxton's share: $3,500

**Answer: B**

*Source: mmlu_pro_business | Category: general | ID: general_028*

---

## Q189. On September 1, Mr. Blake received a statement for his checking account. The closing balance on the statement was $1,810.50. Mr. Blake's checkbook shows a balance of $1,685.75. In comparing his check stubs to the statement, he notices that checks for amounts of $60.80, $40.30, and $25.00 did not appear on the statement. Also, the statement lists a service charge of $1.35 which does not appear on his checkbook stubs. Prepare a reconciliation statement for. Mr. Blake.

- **A.** $1,748.60
- **B.** $1810.50
- **C.** $1,773.00
- **D.** $126.10
- **E.** $1,729.55
- **F.** $1,823.85
- **G.** $1684.40
- **H.** $1,710.05
- **I.** $1685.75
- **J.** $1,654.40

**Answer: G**

*Source: mmlu_pro_business | Category: general | ID: general_029*

---

## Q190. The Cool Hand Luke Corporation adopted the dollar-value LIPO method of inventory evaluation. The price indices were computed using 1969 as the base year. The end of year inventory for each year and the price-level indices are: Inventory at Year-End Prices Price-Level Index Dec. 31, 1969 $16,400 100% Dec. 31, 1970 $16,200 96 Dec. 31, 1971 $20,900 104 Dec. 31, 1972 $26,400 110 Dec. 31, 1973 $24,035 115 Dec. 31, 1974 $26,568 108 Change the current ending inventory cost for 1974 to dollar-value LIFO cost.

- **A.** $20,900
- **B.** $25,086
- **C.** $18,400
- **D.** $27,000
- **E.** $21,200
- **F.** $23,950
- **G.** $19,800
- **H.** $16,875
- **I.** $24,600
- **J.** $22,500

**Answer: B**

*Source: mmlu_pro_business | Category: general | ID: general_030*

---

## Q191. From the very beginning, I wrote to explain my own life to myself, and I invited any readers who chose to make the journey with me to join me on the high wire. I would work without a net and without the noise of the crowd to disturb me. The view from on high is dizzying, instructive. I do not record the world exactly as it comes to me but transform it by making it pass through a prism of fabulous stories I have collected on the way. I gather stories the way a lepidopterist hoards his chloroformed specimens of rare moths, or Costa Rican beetles. Stories are like vessels I use to interpret the world to myself. ----------Pat Conroy Which of the following best describes the organization of the passage?

- **A.** The author explains the importance of storytelling in understanding the world.
- **B.** The author narrates a series of events without providing any interpretation.
- **C.** The author contrasts his methods of work with others in his profession.
- **D.** The author describes a sequence of events leading to his current work.
- **E.** The author uses analogies to explain his experience of a particular action.
- **F.** The author discusses the impact of his profession on his personal life.
- **G.** The author chronicles the various phases of his work in a particular discipline.
- **H.** The author uses his personal experiences to critique societal norms.
- **I.** The author makes a comparison between his own experiences and that of others in his profession.
- **J.** The author provides several explanations for taking a certain course of action.

**Answer: E**

*Source: mmlu_pro_other | Category: general | ID: general_031*

---

## Q192. Gulde’s tax basis in Chyme Partnership was $26,000 at the time Gulde received a liquidating distribution of $12,000 cash and land with an adjusted basis to Chyme of $10,000 and a fair market value of $30,000. Chyme did not have unrealized receivables, appreciated inventory, or properties that had been contributed by its partners. What was the amount of Gulde’s basis in the land?

- **A.** $28,000
- **B.** $30,000
- **C.** $14,000
- **D.** $24,000
- **E.** $10,000
- **F.** $26,000
- **G.** $20,000
- **H.** $0
- **I.** $16,000
- **J.** $12,000

**Answer: C**

*Source: mmlu_pro_other | Category: general | ID: general_032*

---

## Q193. Heavenly Flights charter club charges its members $200 annually. The club's director is considering reducing the annual fee by $2 for all members whenever applicants in excess of 60 members join . For example, if club membership stands at 60 and two new members are added, this will decrease everyone's fee by $4 and the new annual fee would be $196 per member. How many extra members maximize revenue?

- **A.** 110
- **B.** 60
- **C.** 120
- **D.** 50
- **E.** 80
- **F.** 100
- **G.** 130
- **H.** 40
- **I.** 90
- **J.** 70

**Answer: E**

*Source: mmlu_pro_business | Category: general | ID: general_033*

---

## Q194. A passage Jane Eyre is as follows. This was all the account I got from Mrs. Fairfax of her employer and mine. There are people who seem to have no notion of sketching a character, or observing and describing salient points, either in persons or things: the good lady evidently belonged to this class; my queries puzzled, but did not draw her out. Mr. Rochester was Mr. Rochester in her eyes, a gentleman, a landed proprietor‚Äìnothing more: she inquired and searched no further, and evidently wondered at my wish to gain a more definite notion of his identity. The passage suggests that the speaker would describe the "account" mentioned in the first sentence as

- **A.** comprehensive
- **B.** exaggerated
- **C.** insightful
- **D.** precise
- **E.** adequate
- **F.** mystifying
- **G.** deficient
- **H.** erroneous
- **I.** misleading
- **J.** enlightening

**Answer: G**

*Source: mmlu_pro_other | Category: general | ID: general_034*

---

## Q195. The ABC Pencil Co. was considering a price increase and wished to determine the elasticity of demand. An economist and a market researcher, Key and Worce, were hired to study demand. In a controlled experiment, it was determined that at 8\textcent, 100 pencils were sold yielding an elasticity of 2.25. However, key and worce were industrial spies, employed by the EF Pencil Co. And sent to ABC to cause as much trouble as possible. So key and worce decided to change the base for their elasticity figure, measuring price in terms of dollars instead of pennies ( i.e., $.08 for 8\textcent and $.10 for 10\textcent ). How will this sabotage affect the results?

- **A.** The calculated elasticity will appear much higher, misleading the company
- **B.** The change in base will create an illusion of inelastic demand
- **C.** The elasticity measure will decrease
- **D.** The price point for maximum revenue will appear to change dramatically
- **E.** Elasticity is independent of the unit of measurement employed.
- **F.** The price-demand curve will shift to the right
- **G.** The elasticity measure will increase
- **H.** The calculated elasticity will seem artificially low, causing confusion in pricing strategy
- **I.** The sabotage will significantly affect the results
- **J.** The demand will seem more elastic than it actually is

**Answer: E**

*Source: mmlu_pro_economics | Category: general | ID: general_035*

---

## Q196. The managers of Disney World are considering changing the amount charged on their Humpty Dumpty ride. Presently they charge 25 cents a mile and this results in about 6,000 passengers each day. The managers believe the number of daily passengers will rise by 400 for each 1 cent decrease in the admission charge and drop by 400 for each 1 cent increase. What is the admission charge which maximizes the daily revenue?

- **A.** 30 cents
- **B.** 22 cents
- **C.** 18 cents
- **D.** 10 cents
- **E.** 12 cents
- **F.** 25 cents
- **G.** 15 cents
- **H.** 35 cents
- **I.** 20 cents
- **J.** 28 cents

**Answer: I**

*Source: mmlu_pro_business | Category: general | ID: general_036*

---

## Q197. Mr. Ellis sells "BuzzbeeFrisbess\textquotedblright door-to-door. In an average month, he sells 500frisbeesat a price of $5 each. Next month, his company is planning an employee contest whereby if any employee sells 1,000frisbees, he will re-ceive an extra twoweeksvacation with pay. Never one to work too hard, Mr. Ellis decides that instead of trying to push $5 frisbeeson unwilling customers for 12 hours a day, he will maintain his normal work schedule of 8 hours each day. His strategy is to lower the price which he charges his customers. If demand elasticity, e = -3, what price should Mr. Ellis charge in order to sell 1000 "BuzzbeeFrisbees." Use average values for P and Q.

- **A.** $2.50
- **B.** $3.75
- **C.** $4
- **D.** $5
- **E.** $4.25
- **F.** $2
- **G.** $3
- **H.** $5.50
- **I.** $4.5
- **J.** $3.50

**Answer: C**

*Source: mmlu_pro_economics | Category: general | ID: general_037*

---

## Q198. If two variables, $x_t$ and $y_t$ are said to be cointegrated, which of the following statements are true?

i) $x_t$ and $y_t$ must both be stationary


ii) Only one linear combination of $x_t$ and $y_t$ will be stationary


iii) The cointegrating equation for $x_t$ and $y_t$ describes the short-run relationship

between the two series


iv) The residuals of a regression of $y_t$ on $x_t$ must be stationary

- **A.** (i) and (iv) only
- **B.** (iii) and (iv) only
- **C.** (i) and (iii) only
- **D.** (ii) and (iv) only
- **E.** (i), (iii), and (iv) only
- **F.** (i), (ii), and (iii) only
- **G.** (ii) and (iii) only
- **H.** (i) and (ii) only
- **I.** (ii), (iii), and (iv) only
- **J.** (i), (ii), (iii), and (iv)

**Answer: D**

*Source: mmlu_pro_economics | Category: general | ID: general_038*

---

## Q199. Assume that some function, K = F (i) is the cost of the production and marketing of a product, and the total cost (K) is solelydependantupon the number of items produced, whereistands for the number of items produced. Then, the average rate of change in cost = {change in total cost / change in number of items) ; (\DeltaK/\Deltai) = [{F(i_2) - F(i_1)} / {i_2 - i_1}]. Suppose the production and marketing cost of a pair of compasses is K = F (i) = 3\surdi . When O \leqi\leq 250. What then, is the average rate of change in cost, (a) fromi= 25 toi= 100 ? (b) fromi= 25 toi= 225 ?

- **A.** $.15 per unit from 25 to 100 items, $.20 per unit from 25 to 225 items
- **B.** $.10 per unit from 25 to 100 items, $.30 per unit from 25 to 225 items
- **C.** $.40 per unit from 25 to 100 items, $.10 per unit from 25 to 225 items
- **D.** $.25 per unit from 25 to 100 items, $.20 per unit from 25 to 225 items
- **E.** $.30 per unit from 25 to 100 items, $.05 per unit from 25 to 225 items
- **F.** $.22 per unit from 25 to 100 items, $.18 per unit from 25 to 225 items
- **G.** $.35 per unit from 25 to 100 items, $.25 per unit from 25 to 225 items
- **H.** $.18 per unit from 25 to 100 items, $.12 per unit from 25 to 225 items
- **I.** $.20 per unit from 25 to 100 items, $.15 per unit from 25 to 225 items
- **J.** $.25 per unit from 25 to 100 items, $.10 per unit from 25 to 225 items

**Answer: I**

*Source: mmlu_pro_business | Category: general | ID: general_039*

---

## Q200. The three basic reasons for holding money are for a) trans-actions purposes, b) precautionary purposes and c) possible profit or speculation purposes. Given these three motives, indicate which might be most important for the following people; i ) a student on a small allowance from his parents ii) a financially conservative professor with young children to support iii) a "wheeler-dealer"

- **A.** i) Transactions purposes, ii) Transactions purposes, iii) Precautionary reasons
- **B.** i) Precautionary reasons, ii) Precautionary reasons, iii) Transactions purposes
- **C.** i) Speculative purposes, ii) Transactions purposes, iii) Precactionary reasons
- **D.** i) Transactions purposes, ii) Speculative purposes, iii) Precautionary reasons
- **E.** i) Transactions purposes, ii) Precautionary reasons, iii) Speculative purposes
- **F.** i) Speculative purposes, ii) Precautionary reasons, iii) Transactions purposes
- **G.** i) Speculative purposes, ii) Speculative purposes, iii) Precautionary reasons
- **H.** i) Precautionary reasons, ii) Speculative purposes, iii) Transactions purposes
- **I.** i) Precautionary reasons, ii) Transactions purposes, iii) Speculative purposes
- **J.** i) Precautionary reasons, ii) Transactions purposes, iii) Precautionary reasons

**Answer: E**

*Source: mmlu_pro_economics | Category: general | ID: general_040*

---
