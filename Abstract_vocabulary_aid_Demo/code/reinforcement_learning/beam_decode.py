"""
Decoding module for a neural speaker (with attention capabilities).

The MIT License (MIT)
Originally created at 06/15/19, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""
import numpy as np
import torch
import torch.nn.functional as F

def beam_decode(decoder, encoder_out,aux_feat,captions,beam_size,decode_settings):
    temperature=decode_settings['temperature']
    drop_unk=decode_settings['drop_unk']
    drop_bigrams=decode_settings['drop_bigrams']
    remove_bad_endings=decode_settings['remove_bad_endings']
    
    vocab = decoder.vocab
    
    batch_size=len(encoder_out)
    device = encoder_out.device
    enc_image_size = 8

    k = beam_size
    # Tensor to store top k previous words at each step; now they're just <sos>
    k_prev_words = torch.LongTensor([[vocab.sos]*beam_size] * batch_size).view(batch_size,beam_size).to(device)  # (batch_size, 1)=(128,1)

    seq_logprob = torch.ones(batch_size,beam_size,len(vocab)).to(device)  # (batch_size, 1)#128,1
    alpha_init= torch.ones(batch_size,beam_size,1,enc_image_size*enc_image_size).to(device)
    caption_beams_out=[k_prev_words.unsqueeze(-1)]
    log_probs=[seq_logprob]
    alpha_beams_out=[alpha_init]
    seq_mask = torch.ones((batch_size, beam_size, 1), device=device)#128,5,1

    # Start decoding
    step = 0
    if len(captions.shape)>1:
        max_length=captions.shape[1]
    else:
        max_length=50
    
    
    if aux_feat is not None:
        aux_feat_beam=aux_feat.unsqueeze(1).repeat(1,beam_size,1)
    else:
        aux_feat_beam=None
    h, c = decoder.init_hidden_state(encoder_out) #b_s, 512
    encoder_out_beam = encoder_out.unsqueeze(1).repeat(1,beam_size,1,1)
    h = h.unsqueeze(1).repeat(1,beam_size,1)
    c = c.unsqueeze(1).repeat(1,beam_size,1)

    # s (below) is a number less than or equal to k, because sequences are removed
    # from this process once they hit <eos>
    for step in range(max_length-1):
        embeddings = decoder.word_embedding(k_prev_words)  
        #字-图（注意力）
        awe, alpha=get_attention_map(decoder,encoder_out_beam,h,beam_size)
        decoder_input = torch.cat([embeddings, awe], dim=-1)
        if aux_feat_beam is not None:
            decoder_input = torch.cat([decoder_input, aux_feat_beam], dim=-1)
        h, c = decode_step_beam(decoder,decoder_input,h,c) 
        #计算词概率
        scores = decoder.next_word(h) 
        if temperature != 1:
            scores /= temperature
        word_logprob = F.log_softmax(scores, dim=-1) 
        #选择top_k
        candidate = seq_logprob + word_logprob  
        #清洗
        #1.去除unk
        if drop_unk:
            candidate[:,:,vocab.unk] = -99999
        if step==0:
            candidate[:,:,vocab.eos] = -99999
        
        #2.去除两个连词
        if drop_bigrams and step > 0:
            # drop bi-grams with frequency higher than 1.
            prev_usage = caption_beams_out[-1]
            for i_index in range(batch_size):
                for j_index in range(beam_size):
                    candidate[i_index,j_index,prev_usage[i_index,j_index]]=-99999
           
        #去除x and x
        if drop_bigrams and step > 1:
            ## drop x and x
            and_token = decoder.vocab('and')
            prev_usage_1 = caption_beams_out[-1]
            prev_usage_2 = caption_beams_out[-2]
            for i_index in range(batch_size):
                for j_index in range(beam_size):
                    if prev_usage_1[i_index,j_index]==and_token:
                        candidate[i_index,j_index,prev_usage_2[i_index,j_index]]=-99999
        
        
        
        if remove_bad_endings and step>0:
            tmp = np.zeros(candidate.shape)
            prev_bad = np.isin(np.array(k_prev_words.tolist()), decoder.bad_endings_ix)
            # Make it impossible to generate bad_endings
            tmp[:,:,2]=-99999
            tmp=tmp*prev_bad.astype('uint8')[:,:,None]
            tmp=torch.from_numpy(tmp).to(device)
            candidate = candidate + tmp
        
        

        if step>0:
            mask = (word_inds.view(batch_size, beam_size) != vocab.word2idx['<eos>']).float().unsqueeze(-1)
            seq_mask = seq_mask * mask
            word_logprob = word_logprob * seq_mask.expand_as(word_logprob)
            old_seq_logprob = seq_logprob.expand_as(candidate).contiguous()
            old_seq_logprob[:, :, 1:] =-99999
            candidate = seq_mask * candidate + old_seq_logprob * (1 - seq_mask)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 0:
            top_k_scores, top_k_words = candidate[:,0,:].topk(k, 1, True, True)  
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = candidate.view(batch_size,-1).topk(k, 1, True, True)  # (s)
        
        seq_logprob = top_k_scores.unsqueeze(-1)
        # Convert unrolled indices to actual indices of scores
        beams_inds = torch.div(top_k_words, len(vocab), rounding_mode='floor')  
        word_inds = top_k_words % len(vocab)  
        


        #更新：
        seq_mask = torch.gather(seq_mask, 1, beams_inds.unsqueeze(-1))
        h=torch.gather(h, 1, beams_inds.unsqueeze(-1).expand(batch_size, beam_size,h.shape[-1])) 
        c=torch.gather(c, 1, beams_inds.unsqueeze(-1).expand(batch_size, beam_size,c.shape[-1])) 
        k_prev_words=word_inds #(b_s,beam,1)

        alpha_beams_out=list(torch.gather(o, 1, beams_inds.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,o.shape[-1])) for o in alpha_beams_out)
        alpha_beams_out.append(alpha.unsqueeze(2))

        caption_beams_out= list(torch.gather(o, 1, beams_inds.unsqueeze(-1)) for o in caption_beams_out)
        caption_beams_out.append(word_inds.unsqueeze(-1))

        this_word_logprob = torch.gather(word_logprob, 1,beams_inds.unsqueeze(-1).expand(batch_size, beam_size,word_logprob.shape[-1]))
        this_word_logprob = torch.gather(this_word_logprob, 2, word_inds.unsqueeze(-1))

        #记录对数似然
        log_probs = list(
            torch.gather(o, 1, beams_inds.unsqueeze(-1).expand(batch_size, beam_size, 1)) for o in log_probs)
        log_probs.append(this_word_logprob)

    # Sort result
    seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)
    alpha_beams_out=torch.cat(alpha_beams_out,-2)
    alpha_beams_out=torch.gather(alpha_beams_out, 1, sort_idxs.unsqueeze(-1).expand(alpha_beams_out.shape)) 
    
    caption_beams_out = torch.cat(caption_beams_out, -1)
    caption_beams_out = torch.gather(caption_beams_out, 1, sort_idxs.expand(batch_size, beam_size, max_length))
    
    
    log_probs = torch.cat(log_probs, -1)
    log_probs = torch.gather(log_probs, 1, sort_idxs.expand(batch_size, beam_size, max_length))

    return caption_beams_out, log_probs,alpha_beams_out

def decode_step_beam(decoder,decoder_input,h,c):
    h_beam=h.view(-1,h.shape[-1])
    c_beam=c.view(-1,c.shape[-1])
    decoder_input_beam=decoder_input.view(-1,decoder_input.shape[-1])
    h_beam,c_beam=decoder.decode_step(decoder_input_beam, (h_beam, c_beam))
    h=h_beam.view(h.shape)
    c=c_beam.view(c.shape)
    return h,c

def get_attention_map(decoder,encoder_out_beam,h,beam_size):
    batch_size=len(encoder_out_beam)
    dims=encoder_out_beam.shape
    encoder_out_beam_=encoder_out_beam.view(-1,dims[-2],dims[-1])
    h_=h.view(-1,h.shape[-1])

    awe, alpha = decoder.attention(encoder_out_beam_, h_)   
    gate = decoder.sigmoid(decoder.f_beta(h_))  
    awe = gate * awe
    
    awe_=awe.view(batch_size,beam_size,-1)
    alpha_=alpha.view(batch_size,beam_size,-1)
    return awe_,alpha_