"""
The main training function for Ice-Bench

TODO: Need to think about structure for using different models 
- Unet
- DeepLabV3
- ViT

hydra?

"""

import torch
from typing import Optional
from data_processing.ice_data import IceDataset
from monai.losses import DiceLoss

def train(model: torch.nn.Module, train_dataset: IceDataset, val_dataset: Optional[IceDataset] = None, epochs: int = 50, learning_rate: float = 0.0001, optimiser_name: str = "Adam", loss_function_name: str = "DiceLoss"):
    """
    Usual training methodology:
        create the optimiser and loss function
    """

    # define and set up the optimiser

    if optimiser_name == "Adam":
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # dot dot dot
    # all the other optimisers
    '''
    another way you could do this is making an optimiser class or function (just to hold all the if statements so they don't clutter the file)
    model_optimiser = optimiser_class.get(optimiser)
    '''

    if loss_function_name == "DiceLoss":
        loss_function = DiceLoss()
    # same as above move this to a loss class



    for epoch in range(epochs):
        print(epoch)
        # run one epoch of train data
        for image, mask in train_dataset:
            print("new image!")
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)
            # reset gradients
            optimiser.zero_grad()

            # get model predicton
            prediction = model(image)

            
            # find the loss (how good is the prediction)

            # need to check the order, is it good before bad?
            loss = loss_function(prediction, mask)
            print(loss)

            # back propagate the loss
            loss.backward()
            optimiser.step()

            # record the loss if you fancy
        
        # validation!
        if val_dataset is not None:
            for image, mask in val_dataset:
                with torch.no_grad:
                    # get model predicton
                    prediction = model(image)

                    # find the loss (how good is the prediction)

                    # need to check the order, is it good before bad?
                    loss = loss_function(prediction, mask)
                    # save this or do something fun



"""
Main: where model is chosen
"""