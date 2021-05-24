import torch
import torchani
import os
import math
import torch.utils.tensorboard
import tqdm
from torchani.units import hartree2kcalmol
import scipy
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

Rcr = 5.1000e+00
Rca = 3.5000e+00
EtaR = torch.tensor([1.9700000e+01], device=device)
ShfR = torch.tensor([8.0000000e-01, 1.0687500e+00, 1.3375000e+00, 1.6062500e+00, 1.8750000e+00, 2.1437500e+00, 2.4125000e+00, 2.6812500e+00, 2.9500000e+00, 3.2187500e+00, 3.4875000e+00, 3.7562500e+00, 4.0250000e+00, 4.2937500e+00, 4.5625000e+00, 4.8312500e+00], device=device)
Zeta = torch.tensor([1.4100000e+01], device=device)
ShfZ = torch.tensor([3.9269908e-01, 1.1780972e+00, 1.9634954e+00, 2.7488936e+00], device=device)
EtaA = torch.tensor([1.2500000e+01], device=device)
ShfA = torch.tensor([8.0000000e-01, 1.1375000e+00, 1.4750000e+00, 1.8125000e+00, 2.1500000e+00, 2.4875000e+00, 2.8250000e+00, 3.1625000e+00], device=device)
species_order = ['H','C','N','O','F','P','S','Cl']
num_species = len(species_order)
aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
energy_shifter = torchani.utils.EnergyShifter(None)


try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()
dspath = os.path.join(path, 'all_molecules.h5') #dataset
batch_size = 100 #more
print('DATA: ', dspath, '\n')


training, validation = torchani.data.load(dspath).subtract_self_energies(energy_shifter, species_order).species_to_indices(species_order).shuffle().split(0.75, None)

training = training.collate(batch_size).cache()
validation = validation.collate(batch_size).cache()

print('Self atomic energies: ', energy_shifter.self_energies)

aev_dim = aev_computer.aev_length

H_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 256),
    torch.nn.CELU(0.1),
    torch.nn.Linear(256, 192),
    torch.nn.CELU(0.1),
    torch.nn.Linear(192, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 1)
)

C_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 224),
    torch.nn.CELU(0.1),
    torch.nn.Linear(224, 192),
    torch.nn.CELU(0.1),
    torch.nn.Linear(192, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 1)
)

N_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 192),
    torch.nn.CELU(0.1),
    torch.nn.Linear(192, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 1)
)

O_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 192),
    torch.nn.CELU(0.1),
    torch.nn.Linear(192, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 1)
)

F_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

P_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

S_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

Cl_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)


nn = torchani.ANIModel([H_network, C_network, N_network, O_network, S_network, F_network, Cl_network, P_network])
print(nn)

def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        torch.nn.init.zeros_(m.bias)

nn.apply(init_params)

model = torchani.nn.Sequential(aev_computer, nn).to(device)

AdamW = torch.optim.AdamW([
    # H networks
    {'params': [H_network[0].weight], 'weight_decay': 0.005},
    {'params': [H_network[2].weight], 'weight_decay': 0.000001},
    {'params': [H_network[4].weight], 'weight_decay': 0.000001},
    {'params': [H_network[6].weight]},
    # C networks
    {'params': [C_network[0].weight], 'weight_decay': 0.005},
    {'params': [C_network[2].weight], 'weight_decay': 0.000001},
    {'params': [C_network[4].weight], 'weight_decay': 0.000001},
    {'params': [C_network[6].weight]},
    # N networks
    {'params': [N_network[0].weight], 'weight_decay': 0.005},
    {'params': [N_network[2].weight], 'weight_decay': 0.000001},
    {'params': [N_network[4].weight], 'weight_decay': 0.000001},
    {'params': [N_network[6].weight]},
    # O networks
    {'params': [O_network[0].weight], 'weight_decay': 0.005},
    {'params': [O_network[2].weight], 'weight_decay': 0.000001},
    {'params': [O_network[4].weight], 'weight_decay': 0.000001},
    {'params': [O_network[6].weight]},
    # F networks
    {'params': [F_network[0].weight], 'weight_decay': 0.005},
    {'params': [F_network[2].weight], 'weight_decay': 0.000001},
    {'params': [F_network[4].weight], 'weight_decay': 0.000001},
    {'params': [F_network[6].weight]},
    # P networks
    {'params': [P_network[0].weight], 'weight_decay': 0.005},
    {'params': [P_network[2].weight], 'weight_decay': 0.000001},
    {'params': [P_network[4].weight], 'weight_decay': 0.000001},
    {'params': [P_network[6].weight]},
    # S networks
    {'params': [S_network[0].weight], 'weight_decay': 0.005},
    {'params': [S_network[2].weight], 'weight_decay': 0.000001},
    {'params': [S_network[4].weight], 'weight_decay': 0.000001},
    {'params': [S_network[6].weight]},
    # Cl networks
    {'params': [Cl_network[0].weight], 'weight_decay': 0.005},
    {'params': [Cl_network[2].weight], 'weight_decay': 0.000001},
    {'params': [Cl_network[4].weight], 'weight_decay': 0.000001},
    {'params': [Cl_network[6].weight]},
])

SGD = torch.optim.SGD([
    # H networks
    {'params': [H_network[0].bias]},
    {'params': [H_network[2].bias]},
    {'params': [H_network[4].bias]},
    {'params': [H_network[6].bias]},
    # C networks
    {'params': [C_network[0].bias]},
    {'params': [C_network[2].bias]},
    {'params': [C_network[4].bias]},
    {'params': [C_network[6].bias]},
    # N networks
    {'params': [N_network[0].bias]},
    {'params': [N_network[2].bias]},
    {'params': [N_network[4].bias]},
    {'params': [N_network[6].bias]},
    # O networks
    {'params': [O_network[0].bias]},
    {'params': [O_network[2].bias]},
    {'params': [O_network[4].bias]},
    {'params': [O_network[6].bias]},
    # F networks
    {'params': [F_network[0].bias]},
    {'params': [F_network[2].bias]},
    {'params': [F_network[4].bias]},
    {'params': [F_network[6].bias]},
    # P networks
    {'params': [P_network[0].bias]},
    {'params': [P_network[2].bias]},
    {'params': [P_network[4].bias]},
    {'params': [P_network[6].bias]},
    # S networks
    {'params': [S_network[0].bias]},
    {'params': [S_network[2].bias]},
    {'params': [S_network[4].bias]},
    {'params': [S_network[6].bias]},
    # Cl networks
    {'params': [Cl_network[0].bias]},
    {'params': [Cl_network[2].bias]},
    {'params': [Cl_network[4].bias]},
    {'params': [Cl_network[6].bias]},
], lr=1e-3)


AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=100, threshold=0)
SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=0.5, patience=100, threshold=0)

latest_checkpoint = 'new_test_latest.pt' #checkpoint


if os.path.isfile(latest_checkpoint):
    checkpoint = torch.load(latest_checkpoint)
    nn.load_state_dict(checkpoint['nn'])
    AdamW.load_state_dict(checkpoint['AdamW'])
    SGD.load_state_dict(checkpoint['SGD'])
    AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])
    SGD_scheduler.load_state_dict(checkpoint['SGD_scheduler'])


def validate():
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    count = 0
    for properties in validation:
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float()
        true_energies = properties['energies'].to(device).float()
        _, predicted_energies = model((species, coordinates))
        total_mse += mse_sum(predicted_energies, true_energies).item()
        count += predicted_energies.shape[0]
    return hartree2kcalmol(math.sqrt(total_mse / count))

tensorboard = torch.utils.tensorboard.SummaryWriter()

mse = torch.nn.MSELoss(reduction='none')

print("training starting from epoch", AdamW_scheduler.last_epoch + 1)
max_epochs = 2000
early_stopping_learning_rate = 1.0E-5
best_model_checkpoint = 'new_test_best.pt' #best model

for _ in range(AdamW_scheduler.last_epoch + 1, max_epochs):
    rmse = validate()
    print('RMSE:', rmse, 'at epoch', AdamW_scheduler.last_epoch + 1)
    
    learning_rate = AdamW.param_groups[0]['lr']

    if learning_rate < early_stopping_learning_rate:
        break

    # checkpoint
    if AdamW_scheduler.is_better(rmse, AdamW_scheduler.best):
        torch.save(nn.state_dict(), best_model_checkpoint)

    AdamW_scheduler.step(rmse)
    SGD_scheduler.step(rmse)

    tensorboard.add_scalar('validation_rmse', rmse, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('best_validation_rmse', AdamW_scheduler.best, AdamW_scheduler.last_epoch)
    tensorboard.add_scalar('learning_rate', learning_rate, AdamW_scheduler.last_epoch)
    
    for i, properties in tqdm.tqdm(
        enumerate(training),
        total=len(training),
        desc="epoch {}".format(AdamW_scheduler.last_epoch)
    ):
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float()
        true_energies = properties['energies'].to(device).float()
        num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
        _, predicted_energies = model((species, coordinates))

        loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
        
        AdamW.zero_grad()
        SGD.zero_grad()
        loss.backward()
        AdamW.step()
        SGD.step()

        # write current batch loss to TensorBoard
        tensorboard.add_scalar('batch_loss', loss, AdamW_scheduler.last_epoch * len(training) + i)

    torch.save({
        'nn': nn.state_dict(),
        'AdamW': AdamW.state_dict(),
        'SGD': SGD.state_dict(),
        'AdamW_scheduler': AdamW_scheduler.state_dict(),
        'SGD_scheduler': SGD_scheduler.state_dict(),
    }, latest_checkpoint)

def correlation():
    true_e = []
    pred_e =[]
    for properties in validation:
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float()
        true_energies = properties['energies'].to(device).float()
        _, predicted_energies = model((species, coordinates))
        true_e.append(true_energies)
        pred_e.append(predicted_energies)
    true_e = torch.cat((true_e[:])).cpu().detach().numpy()
    pred_e = torch.cat((pred_e[:])).cpu().detach().numpy()
    
    np.savetxt('validation_results.txt', np.column_stack((true_e, pred_e)), delimiter=',')
    
    r, p = scipy.stats.pearsonr(true_e, pred_e)
    print('r is', r)
    print('p-value is', p)
    print('r2 is', r2_score(true_e, pred_e))
    return r, p
    
correlation = correlation()

print("The best mse is", AdamW_scheduler.best)

print('done')
