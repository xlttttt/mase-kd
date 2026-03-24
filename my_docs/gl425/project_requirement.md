  
**ADLS 2026 Team Project Requirements**

**Objective** 

The overall goal is to develop an automated system that can perform a sequence of optimizations to ideally a set of ML models. We accept two streams (software and hardware) of projects, and the corresponding requirements are slightly different.

 The minimum functional requirements for ***software stream projects*** are: 

* Perform optimization of ML models, either at compile-time or run-time, either algorithmic optimizations (eg. pruning or NAS) or implementation optimizations (custom kernels). These optimizations should be integrated in an automated tool-flow.  
* Demonstrate optimization results, demonstrate any potential gains and trade-offs (if any) on the evaluated datasets.   
* Ensure that these optimizations are "principled"—they should adhere to methodologies outlined in published paper or be pertinent to course material. You can provide novel methods or ideas (encouraged, extra marks awarded), but then an established comparison to existing methods are expected. While we permit the incorporation of existing libraries and tools, simplistic or unsophisticated integrations will result in lower marks.  
* Adhere to open-source engineering best practices. Final code should be submitted via a Pull Request (if on MASE), while standalone projects must include detailed READMEs and thorough testing (eg. unit-testing).

The minimum functional requirements for ***hardware stream projects*** are: 

* Hardware implementations for at least three (or four) neural network building blocks (eg. different activation functions).    
* Ensure that there is an automated toolflow to connect different implemented modules (eg. like EmitVerilog on MASE).Evaluation on the resource and performance of the implemented hardware. ‘Show and demonstrate that optimizations and trade-offs of the hardware implementation have been considered.   
* Adhere to open-source engineering best practices. Final code should be submitted via a Pull Request (if on MASE), while standalone projects must include detailed READMEs and thorough testing (eg. unit-testing).

   
The project is open-ended, and it is your decision the detailed functionality of the system. A good approach is to start designing and developing a system for which it is easy to meet the above functional requirements, but also it has space for extensions.

**Coursework deliverables** 

Your coursework deliverables consist of the following: 

- A report (pdf) that describes your system, consisting of at most 4 A4 pages. The report should cover:   
1. The purpose of your system/design.   
2. The overall architecture of your system.   
3. A description of the performance metrics of your system. Which metrics should be used? Why?   
4. At least one diagram of your system’s architecture.   
5. Design decisions taken when implementing the system.   
6. The approach taken to test your system and results.

- Code submission via a PR or sharing your Github Repository.

- Peer feedback: individual submission by each group member to provide peer feedback on your team members, submitted via Microsoft Forms.   
   

**Assessment** 

The coursework mark comes from the following components: 

1. Functionality (40%): does your system work? This is assessed purely based on whether the various parts of the system are functionally correct, and they meet the minimum functional requirements described above.   
      
2. Testing and Engineering Standards (25%) : Is your testing complete? Have you considered testing all aspects of the system? Is your code high-quality, modular and easy to understand?   
      
3. X-factor (25%) : This component aims to capture how challenging your system is and the optimisations that you have applied/considered. For example, when it comes to optimizations, is this low-level CUDA optimization that considers detail such as memory reuse? For hardware projects, do they consider different possible techniques to   
   drive up the clock frequency?   
4. Documentation (10%) : Are the architectural and testing approached adequately described? Have the required components been covered?   
     
5. Peer-feedback (+-10%) : allocated according to peer feedback within the group. This will affect the individual mark by up to 10% relative to the group mark. 

**Submission** 

Submission will be through emailing [a.zhao@imperial.ac.uk](mailto:a.zhao@imperial.ac.uk) the corresponding report and its code url (eg. PR url or repository url).

