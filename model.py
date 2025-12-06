import torch
import torch.nn as nn

class EcomDFCL(nn.Module) :
    def __init__(self):
        super(EcomDFCL,self).__init__()


        self.user_tower = nn.Sequential(
            nn.Linear(12, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.task_tower_treatment_True_revenue = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

        self.task_tower_treatment_False_revenue = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

        self.task_tower_treatment_True_cost = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
        
        self.task_tower_treatment_False_cost = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

    def forward(self, features) :
        """
        Input [batch_size, 12]: 12 features of a (batch of) user(s)
        Output [batch_size, 4]: The predicted values of its (their) cost and revenue either treated or untreated.

        In the output, 
            1st column: treated cost;
            2nd column: treated revenue;
            3rd column: untreated cost;
            4th column: untreated revenue.
        """
        user_feature = self.user_tower(features)

        treated_cost = self.task_tower_treatment_True_cost(user_feature)
        treated_revenue = self.task_tower_treatment_True_revenue(user_feature)
        untreated_cost = self.task_tower_treatment_False_cost(user_feature)
        untreated_revenue = self.task_tower_treatment_False_revenue(user_feature)

        output = torch.cat([
            treated_cost,
            treated_revenue,
            untreated_cost,
            untreated_revenue
        ],dim=1)

        return output
