from sklearn.metrics import confusion_matrix
import torch
import numpy as np
import pandas as pd
def conf_mat(dataloaders, model_ft, nb_classes, file_name):
        
        predlist=torch.zeros(0,dtype=torch.long, device='cpu')
        lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

        with torch.no_grad():
            for i, (inputs, classes) in enumerate(dataloaders['val']):
                inputs = inputs.to(device)
                classes = classes.to(device)
                outputs = model_ft(inputs)
                _, preds = torch.max(outputs, 1)

                # Append batch prediction results
                predlist=torch.cat([predlist,preds.view(-1).cpu()])
                lbllist=torch.cat([lbllist,classes.view(-1).cpu()])

        # Confusion matrix

        conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
        con_mat = np.asarray(conf_mat)
        acc = np.trace(con_mat) / float(np.sum(con_mat))
        print(acc)
        df_cm = pd.DataFrame(con_mat_norm, range(nb_classes), range(nb_classes))
        df_cm.to_csv(file_name)
