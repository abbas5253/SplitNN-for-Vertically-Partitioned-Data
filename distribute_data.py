class Distribute_MNIST:
    """
  This class distribute each image among different workers
  It returns a dictionary with key as data owner's id and 
  value as pointer to the list of data batches at owner's 
  location.
  
  example:-  
  >>> from distribute_data import Distribute_MNIST
  >>> obj = Distribute_MNIST(data-owners= (alice, bob, claire), data_loader= torch.utils.data.DataLoader(trainset)) 
  >>> obj.data_pointer['alice'][1].shape, obj.data_pointer['bob'][1].shape, obj.data_pointer['claire'][1].shape
   (torch.Size([64, 1, 9, 9]),
    torch.Size([64, 1, 9, 9]),
    torch.Size([64, 1, 10, 10]))
  """

    def __init__(self, data_owners, data_loader):

        """
         Args:
          data_owners: tuple of data owners
          data_loader: torch.utils.data.DataLoader for MNIST 

        """

        self.data_owners = data_owners
        self.data_loader = data_loader
        self.no_of_owner = len(data_owners)

        self.data_pointer = {}
        """
        self.data_pointer:  (key, value) = (id of the data holder, a pointer to the list of batches at that data holder).
        """

        # initialize the values of each worker with an empty list
        for onwer in self.data_owners:
            self.data_pointer[onwer.id] = []

        # iterate over each batch of dataloader for, 1) spliting image 2) sending to VirtualWorker
        for images, labels in self.data_loader:

            # calculate width and height according to the no. of workers for UNIFORM distribution
            width, height = [x // (self.no_of_owner) for x in images.shape[2:4]]

            # iterate over each worker for distribution of current batch of the self.data_loader
            for i, owner in enumerate(self.data_owners[:-1]):

                # split the image and send it to VirtualWorker (which is supposed to be a dataowner or client)
                image_part_ptr = images[
                    :, :, width * i : width * (i + 1), height * i : height * (i + 1)
                ].send(owner)

                # append the pointer to the splitted image into data_pointer specified by owner's id as key
                self.data_pointer[owner.id].append(image_part_ptr)

            # Repeat same for the remaining part of the image
            last_owner = self.data_owners[-1]
            last_image_ptr = images[:, :, width * (i + 1) :, height * (i + 1) :].send(
                last_owner
            )
            self.data_pointer[last_owner.id].append(last_image_ptr)
