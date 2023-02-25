def encoder_fuse(encoder_out, feature_map_split, mode):
    batch_size = encoder_out.size(0)
    encoder_dim = encoder_out.size(-1)

    # Flatten image
    encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
    feature_map_split = feature_map_split.permute(0, 2, 3, 1).reshape(batch_size, -1, 1)  # (batch_size, num_pixels, encoder_dim)
    if mode=='cat':
        import torch
        encoder_out=torch.cat([encoder_out,feature_map_split],dim=-1)# b_cX64X513
        encoder_dim=encoder_dim+feature_map_split.size(-1)
    elif mode=='add':
        encoder_out=encoder_out+feature_map_split
    elif mode=='add_cat':
        encoder_out = encoder_out+feature_map_split
        encoder_out=torch.cat([encoder_out,feature_map_split],dim=-1)# b_cX64X513
        encoder_dim=encoder_dim+feature_map_split.size(-1)
    return encoder_out,encoder_dim