"""
# Author Information
======================
Authors: Aparajith Srinivasan & Cedric Manouan 
(CMU 11-785 Introduction to Deep LEarning Teaching Assistants)

Last Update: 16 Dec, 2023

# Code Description
======================
Description: A re-implementation of the Speller in Listen, Attend and Spell (LAS)
Reference: https://arxiv.org/pdf/1508.01211.pdf

"""
import config

import torch 
import torch.nn as nn

class AttentionModule(torch.nn.Module):
    def __init__(self, listener_hidden_size, speller_output_size, projection_size):
        super().__init__()

        self.projection_size    = projection_size

        self.key_projection     = torch.nn.Linear(listener_hidden_size*2, projection_size)
        self.value_projection   = torch.nn.Linear(listener_hidden_size*2, projection_size)
        self.query_projection   = torch.nn.Linear(speller_output_size, projection_size) # because final LSTM Cell has key_value_size output
        #self.context_projection = torch.nn.Linear(speller_output_size, projection_size)
        self.softmax            = torch.nn.Softmax(dim= 1)
  
    def set_key_value(self, encoder_outputs):
        self.key    = self.key_projection(encoder_outputs)
        self.value  = self.value_projection(encoder_outputs)
       
    def compute_context(self, decoder_context):
        
        self.query          = self.query_projection(decoder_context)
        raw_weights         = torch.bmm(self.key, self.query.unsqueeze(2)).squeeze(2) # energy dim (B, T)

        attention_weights   = self.softmax(raw_weights/np.sqrt(self.projection_size))
        attention_context   = torch.bmm(attention_weights.unsqueeze(1), self.value).squeeze(1)

        return attention_context, attention_weights
    
    
class RNNDecoder(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embedding_size,
                 speller_hidden_size,
                 speller_output_size, 
                 speller_lstm_cells,
                 attender:Attention):
        super().__init__()

        self.attention_module         = attender # Attention object in speller
        self.max_timesteps  = config.

        self.embedding      = torch.nn.Embedding(vocab_size, embedding_size, padding_idx= PAD_TOKEN)
        self.emb_dropout    = torch.nn.Dropout(p= config['speller']['embedding_dp'])
        
        self.lstm_cells     = MyLSTMCell(
            input_size      = embedding_size+attender.projection_size,
            hidden_size     = speller_hidden_size,
            output_size     = speller_output_size,
            num_layers      = speller_lstm_cells,
            dropout         = 0.3
        )
        self.gumbel_softmax = LearnableGumbelSoftmax()
        
        # For CDN (Feel free to change)
        #self.CDN_dropout        = torch.nn.Dropout(config['speller']['lstm_dp']/2)
        self.char_prob          = torch.nn.Linear(attender.projection_size+speller_output_size, vocab_size)
        self.char_prob.weight   = self.embedding.weight
        # Weight tying (From embedding layer)

    
    def CDN(self, output_context):
        output_char_dist    = self.char_prob(output_context)
        return output_char_dist
    
    def forward(self, y= None, teacher_forcing_ratio=1):

        batch_size, input_timesteps, _  = self.attention_module.key.shape

        attention_context   = torch.zeros((batch_size, self.attention_module.projection_size)).to(DEVICE)
        output_symbol       = torch.full((batch_size, ), fill_value= SOS_TOKEN).to(DEVICE)
        raw_outputs         = []  
        attention_plot      = []

        if y is None:
            timesteps               = self.max_timesteps
            teacher_forcing_ratio   = 0 #Why does it become zero?

        else:
            timesteps           = y.shape[1] # How many timesteps are we predicting for?
            #label_embeddings    = self.embedding(y)

        hidden_states   = [None]*len(self.lstm_cells)

        for t in range(timesteps):
            p = np.random.random_sample()

            if p < teacher_forcing_ratio and t > 0:
                output_symbol = y[:,t-1]

            char_embed      = self.embedding(output_symbol)
            char_embed      = self.emb_dropout(char_embed)

            lstm_input      = torch.cat([char_embed, attention_context], dim=1)

            lstm_output, hidden_states = self.lstm_cells(lstm_input, hidden_states) # Feed the input through LSTM Cells and attention.
      
            # What should we retrieve from forward_step to prepare for the next timestep?
            attn_context, attn_weights = self.attention_module.compute_context(lstm_output) # Feed the resulting hidden state into attention

            final_context   = torch.cat([self.attention_module.query, attn_context], dim=1)
            raw_pred        = self.CDN(final_context)

            # Generate a prediction for this timestep and collect it in output_symbols
            raw_pred        = self.gumbel_softmax(raw_pred)
            output_symbol   = raw_pred.argmax(dim= 1)

            raw_outputs.append(raw_pred) # for loss calculation
            attention_plot.append(attn_weights) # for plottingn attention plot

        attention_plot  = torch.stack(attention_plot, dim=1)
        raw_outputs     = torch.stack(raw_outputs, dim=1)

        return raw_outputs, attention_plot