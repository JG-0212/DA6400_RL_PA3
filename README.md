# DA6400_RL_PA3

## Submitted by
Jayagowtham J ME21B078

Lalit Jayanti ME21B096

## Contributions

Jayagowtham J ME21B078

  - Implementation of SMDP Q-Learning (equal contribution)
  - Implementation of Intra-Option Q-Learning (equal contribution)
  - Coming up with and implementing alternate options (equal contribution)
  - Visualization and Plotting (equal contribution)
  - Report writing (equal contribution)

Lalit Jayanti ME21B096

  - Implementation of SMDP Q-Learning (equal contribution)
  - Implementation of Intra-Option Q-Learning (equal contribution)
  - Coming up with and implementing alternate options (equal contribution)
  - Visualization and Plotting (equal contribution)
  - Report writing (equal contribution)

## File Structure 

```python
.
├── assets
├── README.md
├── scripts
│   ├── helpers.py
│   ├── intraoption_agent.py                # Intra-Option Q-Learning Agent
│   ├── options.py                          # Implementation for Options
│   ├── policies.py                         # Policies (GoTo, PickUp, DropOff)
│   ├── smdp_agent.py                       # SMDP Q-Learning Agent
│   ├── taxi_utils.py
│   ├── taxi_visualizer.py
│   └── training.py
├── intraoption_option_set_1_training.ipynb # Intra-Option Q-Learning
├── intraoption_option_set_2_training.ipynb
├── smdp_option_set_1_training.ipynb        # SMDP Q-Learning
├── smdp_option_set_2_training.ipynb
└── visualizations.ipynb           # Reward plots and Q-Table visualization
```
## Basic usage
- ```pip install -r requirements.txt```
- To analyze results, fill the hyperparameters in the second cell and run
  - SMDP Q-Learning, Option set 1 : [smdp_option_set_1_training.ipynb](smdp_option_set_1_training.ipynb)
  - SMDP Q-Learning, Option set 2 : [smdp_option_set_2_training.ipynb](smdp_option_set_2_training.ipynb)
  - Intra-Option Q-Learning, Option set 1 : [intraoption_option_set_1_training.ipynb](intraoption_option_set_1_training.ipynb)
  - Intra-Option Q-Learning, Option set 2 : [intraoption_option_set_2_training.ipynb](intraoption_option_set_2_training.ipynb)
  - Make sure to select the correct kernel for your system from the top-right corner of your notebook, while running the above notebooks.
  - Make sure to create the folders ```backups/taxi-smdp-option1-plots```, ```backups/taxi-smdp-option2-plots```, ```backups/taxi-intraoption-option1-plots```, ```backups/taxi-intraoption-option2-plots```, to store the results.

## Results
Following are a few visualizations showing the performance of the SMDP and Intra-Option Q-Learning agents.

### SMDP Q-Learning

 <table>
  <tr>
    <td><img src="results/smdp_option1.gif" title="SMDP Q-Learning, Option set 1" style="width: 100%;"/></td>
    <td> <img src="results/smdp_option2.gif" title="SMDP Q-Learning, Option set 2" style="width: 100%;"/></td>
  </tr>
</table>

## Intra-Option Q-Learning

 <table>
  <tr>
    <td><img src="results/intraoption_option1.gif" title="Intra-Option Q-Learning, Option set 1" style="width: 100%;"/></td>
    <td> <img src="results/intraoption_option2.gif" title="Intra-Option Q-Learning, Option set 2" style="width: 100%;"/></td>
  </tr>
</table>