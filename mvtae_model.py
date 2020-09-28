__author__ = "Jakob Aungiers"
__copyright__ = "Copyright 2020, Jakob Aungiers // Altum Intelligence"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "research@altumintelligence.com"

import os
from tqdm import tqdm
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn, from_numpy
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

class MVTAEModel(nn.Module):
    
    def __init__(self,
                 model_save_path,
                 seq_len, in_data_dims,
                 out_data_dims,
                 model_name,
                 hidden_vector_size,
                 hidden_alpha_size,
                 dropout_p,
                 optim_lr):
        super(MVTAEModel, self).__init__()
        self.seq_len = seq_len
        self.in_data_dims = in_data_dims
        self.out_data_dims = out_data_dims
        self.model_save_path = model_save_path
        
        self.model_name = model_name
        self.best_loss = 1e10
        self.best_epoch = None
        if model_save_path:
            self.tb_writer = SummaryWriter(log_dir=f'./tensorboard')
        self.device = 'cuda'
        print('Using', self.device)
        if self.device == 'cuda':
            print('Using GPU:', torch.cuda.get_device_name(self.device))
            
        self.build_model(hidden_vector_size, hidden_alpha_size, dropout_p, optim_lr)
        self.to(self.device)
        self.eval()
        
    def build_model(self, hidden_vector_size, hidden_alpha_size, dropout_p, optim_lr):
        self.dropout = nn.Dropout(p=dropout_p)
        self.encoder = nn.LSTM(input_size=self.in_data_dims, hidden_size=hidden_vector_size, batch_first=True)
        self.decoder = nn.LSTM(input_size=self.encoder.hidden_size, hidden_size=self.encoder.hidden_size, batch_first=False)
        self.decoder_output = nn.Linear(self.encoder.hidden_size, self.out_data_dims)

        self.alpha_hidden_1 = nn.Linear(self.encoder.hidden_size, hidden_alpha_size)
        self.alpha_hidden_2 = nn.Linear(hidden_alpha_size, hidden_alpha_size)
        self.alpha_out = nn.Linear(hidden_alpha_size, 1)
        
        self.loss_decoder = nn.MSELoss()
        self.loss_alpha = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=optim_lr)
        
    def forward(self, x):
        x = x.to(self.device)

        # Encoder
        encoder_out, encoder_hidden = self.encoder(x)
        hidden_state_vector = encoder_hidden[0]
        
        # Decoder
        encoder_hidden_dropout = self.dropout(hidden_state_vector)
        decoder_out, decoder_hidden = self.decoder(encoder_hidden_dropout.repeat(self.seq_len, 1, 1))
        decoder_output = self.decoder_output(decoder_out.transpose(0,1))

        # Alpha
        alpha_hidden_1 = F.relu(self.alpha_hidden_1(hidden_state_vector))
        alpha_hidden_1_dropout = self.dropout(alpha_hidden_1)
        alpha_hidden_2 = F.relu(self.alpha_hidden_2(alpha_hidden_1_dropout))
        alpha_output = self.alpha_out(alpha_hidden_1).squeeze()

        return hidden_state_vector, decoder_output, alpha_output

    def predict(self, x):
        return self(x)
    
    def fit(self, data_loader, epochs, start_epoch=0, verbose=False):
        with open('loss.log', 'w') as flog:
            flog.write('Timestamp,Epoch,Loss\n')
        for i in tqdm(range(start_epoch, epochs), disable=not verbose):
            self.train()  # set model to training mode
            for x_batch, y_batch in data_loader:
                x = x_batch.to(self.device)
                x_inv = x.flip(1) # reversed sequence (dim 1) of x reconstructed on all dimensions
                y = y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                hidden_state_vector, decoder_output, alpha_output = self(x)

                loss_decoder = self.loss_decoder(decoder_output, x_inv)
                loss_alpha = self.loss_alpha(alpha_output, y)
                loss = loss_decoder + loss_alpha
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.5)
                self.optimizer.step()
                
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_epoch = i
                torch.save(self.state_dict(), os.path.join(self.model_save_path, f'{self.model_name}_best.pth'))
            self.tb_writer.add_scalar('loss', loss, i)
            with open('loss.log', 'a') as flog:
                flog.write(f'{datetime.utcnow()},{i},{loss}\n')
        print(f'Best epoch: {self.best_epoch} | loss {self.best_loss}')
