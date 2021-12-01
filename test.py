def test(dataset, data_split, label_split, federation, model, logger, epoch):
    with torch.no_grad():
        metric = Metric()
        model.train(False)
        """ local test """
        for m in range(cfg['num_users']):
            data_loader = make_data_loader({'test': SplitDataset(dataset, data_split[m])})['test']
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['img'].size(0)
                input['label_split'] = torch.tensor(label_split[m])
                input = to_device(input, cfg['device'])
                ##########################################
                anchor_embedding, output = model(input)
                output = anchor_loss(input, output, anchor_embedding, federation.global_anchors)
                ##########################################
                output['loss_total'] = output['loss_total'].mean() if cfg['world_size'] > 1 else output['loss_total']
                evaluation = metric.evaluate(cfg['metric_name']['test']['Local'], input, output)
                logger.append(evaluation, 'test', input_size)
        """ global test """
        data_loader = make_data_loader({'test': dataset})['test']
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['img'].size(0)
            input = to_device(input, cfg['device'])
            ################################################
            anchor_embedding, output = model(input)
            output = anchor_loss(input, output, anchor_embedding, federation.global_anchors)
            ################################################
            output['loss_total'] = output['loss_total'].mean() if cfg['world_size'] > 1 else output['loss_total']
            evaluation = metric.evaluate(cfg['metric_name']['test']['Global'], input, output)
            logger.append(evaluation, 'test', input_size)
        """ logging info """
        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                         'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', cfg['metric_name']['test']['Local'] + cfg['metric_name']['test']['Global'])
    return
