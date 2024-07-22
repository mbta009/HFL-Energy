# Flow of the algorithm
# Client update(t_1) -> Edge Aggregate(t_2) -> Cloud Aggregate(t_3)

from options import args_parser
import random
from tensorboardX import SummaryWriter

import torch
from client import Client, ClientProcessor, ProcessUnitType
from edge import Edge
from config.configloader import ConfigLoader, EnergyConfig
from cloud import Cloud
from datasets.get_data import get_dataloaders, show_distribution
import copy
import numpy as np
from tqdm import tqdm
from models.mnist_cnn import mnist_lenet
from models.cifar_cnn_3conv_layer import cifar_cnn_3conv
from models.cifar_resnet import ResNet18
from models.mnist_logistic import LogisticRegression
import client_selection
import logging
from constant import C_CPU, C_GPU
from utils import split_dataset_by_percentage

from custom_logging.setup_logger import setup_logger

def get_client_class(args, clients):
    client_class = []
    client_class_dis = [[],[],[],[],[],[],[],[],[],[]]
    for client in clients:
        train_loader = client.train_loader
        distribution = show_distribution(train_loader, args)
        label = np.argmax(distribution)
        client_class.append(label)
        client_class_dis[label].append(client.id)
    print(client_class_dis)
    return client_class_dis

def get_edge_class(args, edges, clients):
    edge_class = [[], [], [], [], []]
    for (i,edge) in enumerate(edges):
        for cid in edge.cids:
            client = clients[cid]
            train_loader = client.train_loader
            distribution = show_distribution(train_loader, args)
            label = np.argmax(distribution)
            edge_class[i].append(label)
    print(f'class distribution among edge {edge_class}')

def initialize_edges_iid(num_edges, clients, args, client_class_dis, client_selection_algorithm: client_selection.ClientSelectionAlgorithm, client_selection_logger: logging.Logger):
    """
    This function is specially designed for partiion for 10*L users, 1-class per user, but the distribution among edges is iid,
    10 clients per edge, each edge have 10 classes
    :param num_edges: L
    :param clients:
    :param args:
    :return:
    """
    #only assign first (num_edges - 1), neglect the last 1, choose the left
    edges = []
    p_clients = [0.0] * num_edges
    for eid in range(num_edges):
        if eid == num_edges - 1:
            break
        assigned_clients_idxes = []
        for label in range(10):
        #     0-9 labels in total
            assigned_client_idx = client_selection_algorithm.select_edge_server_clients(eid, 1, {"cids": client_class_dis[label]})
            client_selection_logger.info("assigned to edge", extra={"edge_id": eid, "selected_clients": assigned_client_idx})
            for idx in assigned_client_idx:
                assigned_clients_idxes.append(idx)
            client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
        edges.append(Edge(id = eid,
                          cids=assigned_clients_idxes,
                          shared_layers=copy.deepcopy(clients[0].model.shared_layers)))
        [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
        edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
        p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                        for sample in list(edges[eid].sample_registration.values())]
        edges[eid].refresh_edgeserver()
    #And the last one, eid == num_edges -1
    eid = num_edges - 1
    assigned_clients_idxes = []
    for label in range(10):
        if not client_class_dis[label]:
            print("label{} is empty".format(label))
        else:
            assigned_client_idx = client_class_dis[label]
            for idx in assigned_client_idx:
                assigned_clients_idxes.append(idx)
            client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
    edges.append(Edge(id=eid,
                      cids=assigned_clients_idxes,
                      shared_layers=copy.deepcopy(clients[0].model.shared_layers)))
    [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
    edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
    p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                    for sample in list(edges[eid].sample_registration.values())]
    edges[eid].refresh_edgeserver()
    return edges, p_clients

def initialize_edges_niid(num_edges, clients, args, client_class_dis, client_selection_algorithm: client_selection.ClientSelectionAlgorithm, client_selection_logger: logging.Logger):
    """
    This function is specially designed for partiion for 10*L users, 1-class per user, but the distribution among edges is iid,
    10 clients per edge, each edge have 5 classes
    :param num_edges: L
    :param clients:
    :param args:
    :return:
    """
    #only assign first (num_edges - 1), neglect the last 1, choose the left
    edges = []
    p_clients = [0.0] * num_edges
    label_ranges = [[0,1,2,3,4],[1,2,3,4,5],[5,6,7,8,9],[6,7,8,9,0]]
    for eid in range(num_edges):
        if eid == num_edges - 1:
            break
        assigned_clients_idxes = []
        label_range = label_ranges[eid]
        for i in range(2):
            for label in label_range:
                #     5 labels in total
                if len(client_class_dis[label]) > 0:
                    assigned_client_idx = client_selection_algorithm.select_edge_server_clients(eid, 1, {"cids": client_class_dis[label]})
                    client_selection_logger.info("assigned to edge", extra={"edge_id": eid, "selected_clients": assigned_client_idx})
                    client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
                else:
                    label_backup = 2
                    assigned_client_idx = client_selection_algorithm.select_edge_server_clients(eid, 1, {"cids": client_class_dis[label]})
                    client_selection_logger.info("assigned to edge", extra={"edge_id": eid, "selected_clients": assigned_client_idx})
                    client_class_dis[label_backup] = list(set(client_class_dis[label_backup]) - set(assigned_client_idx))
                for idx in assigned_client_idx:
                    assigned_clients_idxes.append(idx)
        edges.append(Edge(id = eid,
                          cids=assigned_clients_idxes,
                          shared_layers=copy.deepcopy(clients[0].model.shared_layers)))
        [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
        edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
        p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                        for sample in list(edges[eid].sample_registration.values())]
        edges[eid].refresh_edgeserver()
    #And the last one, eid == num_edges -1
    #Find the last available labels
    eid = num_edges - 1
    assigned_clients_idxes = []
    for label in range(10):
        if not client_class_dis[label]:
            print("label{} is empty".format(label))
        else:
            assigned_client_idx = client_class_dis[label]
            for idx in assigned_client_idx:
                assigned_clients_idxes.append(idx)
            client_class_dis[label] = list(set(client_class_dis[label]) - set(assigned_client_idx))
    edges.append(Edge(id=eid,
                      cids=assigned_clients_idxes,
                      shared_layers=copy.deepcopy(clients[0].model.shared_layers)))
    [edges[eid].client_register(clients[client]) for client in assigned_clients_idxes]
    edges[eid].all_trainsample_num = sum(edges[eid].sample_registration.values())
    p_clients[eid] = [sample / float(edges[eid].all_trainsample_num)
                    for sample in list(edges[eid].sample_registration.values())]
    edges[eid].refresh_edgeserver()
    return edges, p_clients

def all_clients_test(server, clients, cids, device):
    [server.send_to_client(clients[cid]) for cid in cids]
    for cid in cids:
        server.send_to_client(clients[cid])
        # The following sentence!
        clients[cid].sync_with_edgeserver()
    correct_edge = 0.0
    total_edge = 0.0
    for cid in cids:
        correct, total = clients[cid].test_model(device)
        correct_edge += correct
        total_edge += total
    return correct_edge, total_edge

def fast_all_clients_test(v_test_loader, global_nn, device):
    correct_all = 0.0
    total_all = 0.0
    with torch.no_grad():
        for data in v_test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = global_nn(inputs)
            _, predicts = torch.max(outputs, 1)
            total_all += labels.size(0)
            correct_all += (predicts == labels).sum().item()
    return correct_all, total_all

def initialize_global_nn(args):
    if args.dataset == 'mnist':
        if args.model == 'lenet':
            global_nn = mnist_lenet(input_channels=1, output_channels=10)
        elif args.model == 'logistic':
            global_nn = LogisticRegression(input_dim=1, output_dim=10)
        else: raise ValueError(f"Model{args.model} not implemented for mnist")
    elif args.dataset == 'cifar10':
        if args.model == 'cnn_complex':
            global_nn = cifar_cnn_3conv(input_channels=3, output_channels=10)
        elif args.model == 'resnet18':
            global_nn = ResNet18()
        else: raise ValueError(f"Model{args.model} not implemented for cifar")
    else: raise ValueError(f"Dataset {args.dataset} Not implemented")
    return global_nn

def configure_parameter_c(clients: list[ClientProcessor]):
    for client in clients:
        if client.processor_type.value == "gpu":
            if client.get_config().dvfs_gpu:
                client.get_config().c_gpu = 0.01
            else:
                client.get_config().c_gpu = random.choice(C_GPU)
        else:
            if client.get_config().dvfs_cpu:
                client.get_config().c_cpu = 0.040
            else:
                client.get_config().c_cpu = random.choice(C_CPU)

def HierFAVG(args):
    #make experiments repeatable
    client_selection_logger = setup_logger("client_selection", "./logs/client_selection.log")
    total_energy_logger = setup_logger("total_energy", "./logs/total_energy.log")
    client_energy_configs = ConfigLoader.load_energy_config(num_clients=args.num_clients)
    client_selection_config = ConfigLoader.load_client_selection_config(client_ids=set([i for i in range(args.num_clients)]), num_edges=args.num_edges, num_rounds=args.num_edge_aggregation)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cuda_to_use = torch.device(f'cuda:{args.gpu}')
    device = cuda_to_use if torch.cuda.is_available() else "cpu"
    print(f'Using device {device}')
    FILEOUT = f"{args.dataset}_clients{args.num_clients}_edges{args.num_edges}_" \
              f"t1-{args.num_local_update}_t2-{args.num_edge_aggregation}" \
              f"_model_{args.model}iid{args.iid}edgeiid{args.edgeiid}epoch{args.num_communication}" \
              f"bs{args.batch_size}lr{args.lr}lr_decay_rate{args.lr_decay}" \
              f"lr_decay_epoch{args.lr_decay_epoch}momentum{args.momentum}"
    writer = SummaryWriter(comment=FILEOUT)
    # Build dataloaders
    train_loaders, test_loaders, v_train_loader, v_test_loader = get_dataloaders(args)
    if args.show_dis:
        for i in range(args.num_clients):
            train_loader = train_loaders[i]
            print(len(train_loader.dataset))
            distribution = show_distribution(train_loader, args)
            print("train dataloader {} distribution".format(i))
            print(distribution)

        for i in range(args.num_clients):
            test_loader = test_loaders[i]
            test_size = len(test_loaders[i].dataset)
            print(len(test_loader.dataset))
            distribution = show_distribution(test_loader, args)
            print("test dataloader {} distribution".format(i))
            print(f"test dataloader size {test_size}")
            print(distribution)
    # initialize clients and server
    clients: list[ClientProcessor] = []
    client_id_iter = process_id_iter = 0    
    while client_id_iter < args.num_clients:
        c = Client(client_id_iter,
                   train_loader=train_loaders[client_id_iter],
                   test_loader=test_loaders[client_id_iter],
                   args=args)        
        if client_energy_configs[client_id_iter].gpu:
            spilted_dataset = split_dataset_by_percentage(train_loaders[client_id_iter].dataset)
            spilted_testset = split_dataset_by_percentage(test_loaders[client_id_iter].dataset)
            for i, v in enumerate(ProcessUnitType):
                print(i, v)
                client_processor = ClientProcessor(id=process_id_iter,
                                                train_loader=spilted_dataset[i],
                                                test_loader=spilted_testset[i],
                                                args=args,
                                                device=device,
                                                processor_type=v,
                                                config=client_energy_configs[client_id_iter],
                                                client=c
                                                )
                c.register_processor(client_processor)
                clients.append(client_processor)
                process_id_iter += 1
        else:
            client_processor = ClientProcessor(id=process_id_iter,
                                               train_loader=train_loaders[client_id_iter],
                                               test_loader=test_loaders[client_id_iter],
                                               args=args,
                                               device=device,
                                               config=client_energy_configs[client_id_iter],
                                               client=c
                                            )
            c.register_processor(client_processor)
            clients.append(client_processor)            
            process_id_iter += 2
        client_id_iter += 1

    initilize_parameters = list(clients[0].model.shared_layers.parameters())
    nc = len(initilize_parameters)
    for client in clients:
        user_parameters = list(client.model.shared_layers.parameters())
        for i in range(nc):
            user_parameters[i].data[:] = initilize_parameters[i].data[:]

    # Check the client selection algorithm
    client_selector = client_selection.PredefinedClientSelection(client_selection_config)

    # Initialize edge server and assign clients to the edge server
    edges = []
    cids = np.arange(len(clients))
    clients_per_edge = int(len(clients) / args.num_edges)
    p_clients = [0.0] * args.num_edges

    if args.iid == -2:
        if args.edgeiid == 1:
            client_class_dis = get_client_class(args, clients)
            edges, p_clients = initialize_edges_iid(num_edges=args.num_edges,
                                                    clients=clients,
                                                    args=args,
                                                    client_class_dis=client_class_dis,
                                                    client_selection_algorithm=client_selector,
                                                    client_selection_logger=client_selection_logger)
        elif args.edgeiid == 0:
            client_class_dis = get_client_class(args, clients)
            edges, p_clients = initialize_edges_niid(num_edges=args.num_edges,
                                                     clients=clients,
                                                     args=args,
                                                     client_class_dis=client_class_dis,
                                                     client_selection_algorithm=client_selector,
                                                     client_selection_logger=client_selection_logger)
    else:
        # This is randomly assign the clients to edges
        for i in range(args.num_edges):
            #Randomly select clients and assign them
            selected_cids = client_selector.select_edge_server_clients(i, clients_per_edge, {"cids": cids})
            client_selection_logger.info("assigned to edge", extra={"edge_id": i, "selected_clients": selected_cid})
            cids = list (set(cids) - set(selected_cids))
            edges.append(Edge(id = i,
                              cids = selected_cids,
                              shared_layers = copy.deepcopy(clients[0].model.shared_layers)))
            [edges[i].client_register(clients[cid]) for cid in selected_cids]
            edges[i].all_trainsample_num = sum(edges[i].sample_registration.values())
            p_clients[i] = [sample / float(edges[i].all_trainsample_num) for sample in
                    list(edges[i].sample_registration.values())]
            edges[i].refresh_edgeserver()
    # Initialize cloud server
    cloud = Cloud(shared_layers=copy.deepcopy(clients[0].model.shared_layers))
    # First the clients report to the edge server their training samples
    [cloud.edge_register(edge=edge) for edge in edges]
    print(cloud.sample_registration, len(clients), args.num_clients, args.gpus)
    p_edge = [sample / sum(cloud.sample_registration.values()) for sample in
                list(cloud.sample_registration.values())]
    cloud.refresh_cloudserver()

    #New an NN model for testing error

    global_nn = initialize_global_nn(args)
    if args.cuda:
        global_nn = global_nn.cuda(device)

    #Begin training
    for num_comm in tqdm(range(args.num_communication)):
        configure_parameter_c(clients)
        for client in clients:
            client.calculate_energy(num_comm)
        cloud.refresh_cloudserver()
        [cloud.edge_register(edge=edge) for edge in edges]
        for num_edgeagg in range(args.num_edge_aggregation):
            edge_loss = [0.0]* args.num_edges
            edge_sample = [0]* args.num_edges
            correct_all = 0.0
            total_all = 0.0
            # no edge selection included here
            # for each edge, iterate
            for i,edge in enumerate(edges):
                edge.refresh_edgeserver()
                client_loss = 0.0
                selected_cnum = max(int(clients_per_edge * args.frac),1)
                selected_cids = client_selector.select_client_for_train(edge, selected_cnum, {"p_clients": p_clients[i]})
                client_selection_logger.info("selected clients", extra={"edge_id": edge.id, "num_comm": num_comm, "num_edgeagg": num_edgeagg, "selected_clients": selected_cids})
                total_energy = 0
                for selected_cid in selected_cids:
                    total_energy += clients[selected_cid].client_energy.total_energy
                total_energy_logger.info(f"Round Energy {num_comm} {edge.id} = {total_energy}")

                for selected_cid in selected_cids:                    
                    edge.client_register(clients[selected_cid])
                for selected_cid in selected_cids:
                    edge.send_to_client(clients[selected_cid])
                    clients[selected_cid].sync_with_edgeserver()
                    client_loss += clients[selected_cid].local_update(num_iter=args.num_local_update,
                                                                      device = device)
                    clients[selected_cid].send_to_edgeserver(edge)
                edge_loss[i] = client_loss
                edge_sample[i] = sum(edge.sample_registration.values())

                edge.aggregate(args)
                correct, total = all_clients_test(edge, clients, edge.cids, device)
                correct_all += correct
                total_all += total
            # end interation in edges
            all_loss = sum([e_loss * e_sample for e_loss, e_sample in zip(edge_loss, edge_sample)]) / sum(edge_sample)
            avg_acc = correct_all / total_all
            writer.add_scalar(f'Partial_Avg_Train_loss',
                          all_loss,
                          num_comm* args.num_edge_aggregation + num_edgeagg +1)
            writer.add_scalar(f'All_Avg_Test_Acc_edgeagg',
                          avg_acc,
                          num_comm * args.num_edge_aggregation + num_edgeagg + 1)

        # Now begin the cloud aggregation
        for edge in edges:
            edge.send_to_cloudserver(cloud)
        cloud.aggregate(args)
        for edge in edges:
            cloud.send_to_edge(edge)

        global_nn.load_state_dict(state_dict = copy.deepcopy(cloud.shared_state_dict))
        global_nn.train(False)
        correct_all_v, total_all_v = fast_all_clients_test(v_test_loader, global_nn, device)
        avg_acc_v = correct_all_v / total_all_v
        writer.add_scalar(f'All_Avg_Test_Acc_cloudagg_Vtest',
                          avg_acc_v,
                          num_comm + 1)

    writer.close()
    print(f"The final virtual acc is {avg_acc_v}")

def main():
    args = args_parser()
    HierFAVG(args)

if __name__ == '__main__':
    main()