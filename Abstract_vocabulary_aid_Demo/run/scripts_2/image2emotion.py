import os
import pprint
import torch
import argparse
import pandas as pd
import os.path as osp
import numpy as np
from ast import literal_eval
from plotly.offline import init_notebook_mode, iplot
import plotly
import plotly.express as px

from RAIVS.artemis_model.in_out.neural_net_oriented import torch_load_model, torch_save_model, save_state_dicts
from RAIVS.artemis_model.in_out.neural_net_oriented import image_emotion_distribution_df_to_pytorch_dataset
from RAIVS.artemis_model.in_out.basics import create_dir
from artemis.emotions import ARTEMIS_EMOTIONS

from RAIVS.original_model.neural_models.mlp import MLP
from RAIVS.original_model.neural_models.resnet_encoder import ResnetEncoder
from RAIVS.original_model.neural_models.image_emotion_clf import ImageEmotionClassifier
from RAIVS.original_model.neural_models.image_emotion_clf import single_epoch_train, evaluate_on_dataset





## helper function.
## to evaluate how well the model does according to the class that it finds most likely
## note it only concerns the predictions on examples (images) with a single -unique maximizer- emotion
def evaluate_argmax_prediction(dataset, guesses):
    labels = dataset.labels
    labels = np.vstack(labels.to_numpy())
    unique_max = (labels == labels.max(1, keepdims=True)).sum(1) == 1
    umax_ids = np.where(unique_max)[0]
    gt_max = np.argmax(labels[unique_max], 1)
    max_pred = np.argmax(guesses[umax_ids], 1)
    return (gt_max == max_pred).mean()



def get_preparedData(image_hists,artemis_data):
    # this literal_eval brings the saved string to its corresponding native (list) type
    image_hists.emotion_histogram = image_hists.emotion_histogram.apply(literal_eval)
    # normalize the histograms
    image_hists.emotion_histogram = image_hists.emotion_histogram.apply(lambda x: (np.array(x) / float(sum(x))).astype('float32'))
    print(f'Histograms corresponding to {len(image_hists)} images')

    ## keep each image once.
    artemis_data = artemis_data.drop_duplicates(subset=['art_style', 'painting'])
    artemis_data.reset_index(inplace=True, drop=True)

    # keep only relevant info + merge
    artemis_data = artemis_data[['art_style', 'painting', 'split']] 
    artemis_data = artemis_data.merge(image_hists)
    artemis_data = artemis_data.rename(columns={'emotion_histogram': 'emotion_distribution'})

    n_emotions = len(image_hists.emotion_histogram[0])
    print('Using {} emotion-classes.'.format(n_emotions))
    assert all(image_hists.emotion_histogram.apply(len) == n_emotions)

    return artemis_data

def init_model(n_emotions):
    img_encoder = ResnetEncoder('resnet34', adapt_image_size=1).unfreeze(level=7, verbose=True)
    img_emb_dim = img_encoder.embedding_dimension()

    # here we make an MLP closing with LogSoftmax since we want to train this net via KLDivLoss
    clf_head = MLP(img_emb_dim, [100, n_emotions], dropout_rate=0.3, b_norm=True, closure=torch.nn.LogSoftmax(dim=-1))

    model = ImageEmotionClassifier(img_encoder, clf_head);
    return  model

def train(model, data_loaders, criterion, optimizer, device):
    # set to True, if you are not using a pretrained model
    max_train_epochs = 25
    no_improvement = 0
    min_eval_loss = np.Inf
    best_model=None

    for epoch in range(1, max_train_epochs+1):
        train_loss = single_epoch_train(model, data_loaders['train'], criterion, optimizer, device)
        print('Train Loss: {:.3f}'.format(train_loss))

        eval_loss, _ = \
        evaluate_on_dataset(model, data_loaders['val'], criterion, device, detailed=False)
        print('Eval Loss: {:.3f}'.format(eval_loss))

        if eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            no_improvement = 0
            print('Epoch {}. Validation loss improved!'.format(epoch))
            torch_save_model(model, checkpoint_file)
                
            test_loss, test_confidence = \
            evaluate_on_dataset(model, data_loaders['test'], criterion, device, detailed=True)
            print('Test Loss: {:.3f}'.format(test_loss))                

            dataset = data_loaders['test'].dataset        
            arg_max_acc = evaluate_argmax_prediction(dataset, test_confidence)
            print('Test arg_max_acc: {:.3f}'.format(arg_max_acc))
            best_model = model
        else:
            no_improvement += 1
        
        if no_improvement >=5 :
            print('Breaking at epoch {}. Since for 5 epoch we observed no (validation) improvement.'.format(epoch))
            break
    return model

def test_and_plot(model,args,data_loaders,criterion,device):
    model = torch_load_model(args.checkpoint_file)    
    test_loss, test_confidence = evaluate_on_dataset(model, data_loaders['test'], criterion, device, detailed=True)

    ## how often the most & second most, predicted emotions are positive vs. negative?
    preds = torch.from_numpy(test_confidence)
    top2 = preds.topk(2).indices
    has_pos = torch.any(top2 <= 3, -1)
    has_neg = torch.any((top2 >=4) & (top2 !=8), -1)
    has_else = torch.any(top2 == 8, -1)
    pn = (has_pos & has_neg).double().mean().item()
    pne = ((has_pos & has_neg) | (has_pos & has_else) | (has_neg & has_else)).double().mean().item()
    print('The classifier finds the 1st/2nd most likely emotions to be negative/positive, or contain something-else')
    print(pn, pne)

    # How well it does on test images that have strong majority in emotions?
    labels = data_loaders['test'].dataset.labels
    labels = np.vstack(labels.to_numpy())

    for use_strong_domi in [True, False]:
        print('use_strong_domi:', use_strong_domi)
        if use_strong_domi:
            dominant_max = (labels.max(1) > 0.5)
        else:
            dominant_max = (labels.max(1) >= 0.5)

        umax_ids = np.where(dominant_max)[0]
        gt_max = np.argmax(labels[dominant_max], 1)
        max_pred = np.argmax(test_confidence[umax_ids], 1)    

        print('Test images with dominant majority', dominant_max.mean())
        print('Guess-correctly', (gt_max == max_pred).mean(), '\n')

    '''
        plot(plot_confusion_matrix(ground_truth=gt_max, predictions=max_pred, labels=ARTEMIS_EMOTIONS))
        fig = px.line(df, x="year", y="lifeExp", color='country')

    # html file
    #plotly.offline.plot(fig, filename='./lifeExp.html')
    fig.write_image("./lifeExp.png")
    print("done")


        # For the curious one. Images where people "together" aggree on anger are rare. Why?
        plot(plot_confusion_matrix(ground_truth=gt_max, predictions=max_pred, labels=ARTEMIS_EMOTIONS, normalize=False))

    '''
def get_args():
    parser = argparse.ArgumentParser(description='testing-image2emoclf')
    # config
    parser.add_argument('--cfg', type=str, default='F:/work/Image_emotion_analysis/artemis-master/RAIVS/configs_local/image2emotion_local.yml',
                    help='configuration; similar to what is used in detectron')
    parser.add_argument(
        '--set_cfgs', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]/n This has higher priority'
             'than cfg file but lower than other args. (You can only overwrite'
             'arguments that have alerady been defined in config file.)',
        default=[], nargs='+')
    #load params from yml
    # step 1: read cfg_fn
    args = parser.parse_args()
    if args.cfg is not None or args.set_cfgs is not None:
        from RAIVS.utils.config import CfgNode
        if args.cfg is not None:
            cn = CfgNode(CfgNode.load_yaml_with_base(args.cfg))
        else:
            cn = CfgNode()
        if args.set_cfgs is not None:
            cn.merge_from_list(args.set_cfgs)
        for k,v in cn.items():
            setattr(args, k, v)
        args = parser.parse_args(namespace=args)

    # pprint them
    print('\nParameters Specified:')
    args_string = pprint.pformat(vars(args))
    print(args_string)
    print('\n')
    return args
    


if __name__=='__main__':

    args = get_args()
    if os.path.exists(args.save_dir):
        create_dir(args.save_dir)
    checkpoint_file = osp.join(args.save_dir, 'best_model.pt')
    # minor parameters
    GPU_ID = 0 
    ## Prepare the artemis dataset (merge it with the emotion-histograms.)
    artemis_data = pd.read_csv(args.artemis_preprocessed_dir)
    image_hists = pd.read_csv(args.image_hists_file)
    print('Annotations loaded:', len(artemis_data))

    artemis_data = get_preparedData(image_hists,artemis_data)
    ## prepare data
    data_loaders, datasets = image_emotion_distribution_df_to_pytorch_dataset(artemis_data, args)

    ## Prepate the Neural-Net Stuff (model, optimizer etc.)
    ## This is what I used for the paper with minimal hyper-param-tuning. You can use different nets/configs here...
    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.KLDivLoss(reduction='batchmean').to(device)
    model = init_model(9).to(device)
    optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': 5e-4}])
    
    print('Starting Training……')
    model = train(model, data_loaders, criterion, optimizer, device)

    test_and_plot(model,args,data_loaders,criterion,device)
    print('done')