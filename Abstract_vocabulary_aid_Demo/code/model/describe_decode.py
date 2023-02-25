import torch
from RAIVS.reinforcement_learning.beam_decode import beam_decode

def encoder_fuse(encoder_out, feature_map_split, mode):
    batch_size = encoder_out.size(0)
    encoder_dim = encoder_out.size(-1)

    # Flatten image
    encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
    feature_map_split = feature_map_split.permute(0, 2, 3, 1).reshape(batch_size, -1, 1)  # (batch_size, num_pixels, encoder_dim)
    '''
    if mode=='cat':
        import torch
        encoder_out=torch.cat([encoder_out,feature_map_split],dim=-1)# b_cX64X513
        encoder_dim=encoder_dim+feature_map_split.size(-1)
    elif mode=='add':
        encoder_out=encoder_out+feature_map_split
    elif mode=='add_cat':
    '''
    encoder_out = encoder_out+feature_map_split
    encoder_out=torch.cat([encoder_out,feature_map_split],dim=-1)# b_cX64X513
    encoder_dim=encoder_dim+feature_map_split.size(-1)
    return encoder_out,encoder_dim


def describe_decode(train_mode, mode,model, image, label, abstract_data, caps_token, decode_settings):
    if train_mode=='scst':
        return scst_decode(mode, model, image, label, abstract_data, caps_token, decode_settings)
    else:
        return xe_decode(mode, model, image, label, abstract_data, caps_token)

def scst_decode(mode,model, image, label,abstract_data, caps_token, decode_settings):
    encoder=None
    if mode=='original_model':
        encoder_out=model.encoder(image)
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
    elif mode=='EMOdetected_model':
        encoder_out=model.encoder(image)
        _,_,emo_area=model.emo_map_encoder(image)
        feature_map_split_new = []
        for id,index in enumerate(label.argmax(dim=1)):
            feature_map_split_new.append(emo_area[id,index,:,:].unsqueeze(0).unsqueeze(0))
        feature_map_split_new = torch.cat(feature_map_split_new,dim=0)

        encoder_out, _ = encoder_fuse(encoder_out, feature_map_split_new, mode)
    beam_size =5
    output = beam_decode(model.decoder, encoder_out,label,abstract_data, caps_token,beam_size,decode_settings)
    return output


def xe_decode(mode, model, image, label, abstract_data,caps_token):
    encoder_out=None
    if mode=='original_model':
        encoder_out=model.encoder(image)
        '''
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        '''
    elif mode=='EMOdetected_model':
        emo_map_encoder=model.emo_map_encoder(image)
        _,_,emo_area=model.emotion_decoder(image)
        feature_map_split_new = []
        for id,index in enumerate(label.argmax(dim=1)):
            feature_map_split_new.append(emo_area[id,index,:,:].unsqueeze(0).unsqueeze(0))
        feature_map_split_new = torch.cat(feature_map_split_new,dim=0)

        encoder_out, _ = encoder_fuse(emo_map_encoder, feature_map_split_new, mode)
        
    output=model.decoder(encoder_out,caps_token,label,abstract_data)
    return output
