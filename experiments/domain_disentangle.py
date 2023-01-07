import torch
from models.base_model import DomainDisentangleModel

def myReconstructorLoss(reconstructorOutputs, features):
    return torch.nn.MSELoss(reconstructorOutputs, features) + torch.nn.KLDivLoss(reconstructorOutputs, features)

def myEntropyLoss(outputs):
    l = 0
    for i in range(len(outputs)):
        l += outputs[i].item()
    l *= 1/len(outputs)
    return -l
    

class DomainDisentangleExperiment: # See point 2. of the project
    
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')
        
        # Setup model
        self.model = DomainDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Setup optimization procedure
        self.object_classifier_optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.object_classifier_criterion = torch.nn.CrossEntropyLoss()

        self.domain_classifier_optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.domain_classifier_criterion = torch.nn.CrossEntropyLoss()

        self.domain_category_optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.domain_category_criterion = myEntropyLoss

        self.object_domain_optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.object_domain_criterion = myEntropyLoss

        self.reconstructor_optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.reconstructor_criterion = myReconstructorLoss

    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss

        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration, best_accuracy, total_train_loss

    def train_iteration(self, data, train=True):
        if train:
            x, y, z = data
            x = x.to(self.device)
            y = y.to(self.device)
            z = z.to(self.device)

            self.object_classifier_optimizer.zero_grad()
            logits = self.model(x, w1=1)
            loss = self.object_classifier_criterion(logits, y)
            loss.backward()
            self.object_classifier_optimizer.step()

            self.domain_classifier_optimizer.zero_grad()
            logits = self.model(x, w2=1)
            loss = self.domain_classifier_criterion(logits, z)
            loss.backward()

            self.domain_category_optimizer.zero_grad()
            logits = self.model(x, w3=1)
            loss = self.domain_category_criterion(logits)
            loss.backward()
            
            self.object_domain_optimizer.zero_grad()
            logits = self.model(x, w4=1)
            loss = self.object_domain_criterion(logits)
            loss.backward()
            self.object_domain_optimizer.step()

            self.reconstructor_optimizer.zero_grad()
            logits = self.model(x, w5=1)
            loss = self.reconstructor_criterion(logits, x)
            loss.backward()

        else:
            x, y, z = data
            x = x.to(self.device)
            z = z.to(self.device)

            logits = self.model(x, w2=1)
            loss = self.domain_classifier_criterion(logits, z)
            loss.backward()
            self.domain_category_optimizer.step()

            logits = self.model(x, w3=1)
            loss = self.domain_category_criterion(logits)
            loss.backward()
            self.domain_category_optimizer.step()

            logits = self.model(x, w5=1)
            loss = self.reconstructor_criterion(logits, x)
            loss.backward()
            self.reconstructor_optimizer.step()

        return loss.item()

    def validate(self, loader):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x, w1=1)
                loss += self.object_classifier_criterion(logits, y)
                pred = torch.argmax(logits, dim=-1)
                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss