from __future__ import print_function

from datetime import datetime
import os
import sys
import time
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
log = logging.getLogger(__name__)

from azure.storage.file import FileService
from azure.storage.blob import BlockBlobService
from azure.mgmt.resource import ResourceManagementClient

from azureml.train.dnn import TensorFlow
from azureml.core.workspace import Workspace
from azureml.core.datastore import Datastore
from azureml.core.compute import ComputeTarget, BatchAiCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.run import Run
#from azureml.core.runconfig import DataReferenceConfiguration
from azureml.core import Experiment
from azureml.train.hyperdrive import *

# utilities.py contains helper functions used by different notebooks
from aml_utils.config import AMLConfiguration

def create_and_attach_blob_storage(cfg, ws):
    """ If required, creates the blob storage containers in the datareferences of cfg """
    if len(cfg.DataReference.localDirectoryBlobList) > 0:
        for ref in cfg.DataReference.localDirectoryBlobList:
            log.info("Attempting to create Blob Container '%s' on storage account '%s'.", ref.remoteBlobContainer, ref.storageAccountName)
            blob_service = BlockBlobService(ref.storageAccountName, ref.storageAccountKey)
            exist = blob_service.create_container(ref.remoteBlobContainer, fail_on_exist=False)
            if exist:
                log.info("Blob Container '%s' on storage account '%s' created.", ref.remoteBlobContainer, ref.storageAccountName)
            else:
                log.info("Blob Container '%s' on storage account '%s' already existed.", ref.remoteBlobContainer, ref.storageAccountName)
            # Get most recent list of datastores linked to current workspace
            datastores = ws.datastores()
            # Validate if ds is created
            ds = None if ref.dataref_id not in datastores else Datastore(workspace = ws, name = ref.dataref_id)
            # If DS exists and isn't mapped to the right place
            if ds:
                if ds.account_name == ref.storageAccountName and ds.container_name == ref.remoteBlobContainer:
                    recreate = False
                else:
                    recreate = True
                    # also remove the existing reference
                    ds.unregister()
            else:
                recreate = True
            if recreate:
                log.info('Registering blob "{}" to AML datastore for AML workspace "{}" under datastore id "{}".'.format(ref.remoteBlobContainer, ws.name, ref.dataref_id))
                ds = Datastore.register_azure_blob_container(workspace = ws,
                                                    datastore_name = ref.dataref_id, 
                                                    container_name = ref.remoteBlobContainer, 
                                                    account_name = ref.storageAccountName, 
                                                    account_key = ref.storageAccountKey,
                                                    overwrite = True,  # Overwrites the datastore (not the data itself, the object) if it already is part of this workspace
                                                    )
            else:
                log.info('Blob "{}" under AML workspace "{}" already registered under datastore id "{}".'.format(ref.remoteBlobContainer, ws.name, ref.dataref_id))

def create_and_attach_file_storage(cfg, ws):
    if len(cfg.DataReference.localDirectoryFilesList) > 0:
        for ref in cfg.DataReference.localDirectoryFilesList:
            log.info("Attempting to create file share '%s' on storage account '%s'.", ref.remoteFileShare, ref.storageAccountName)
            file_service = FileService(ref.storageAccountName, ref.storageAccountKey)
            exist = file_service.create_share(ref.remoteFileShare, fail_on_exist=False)
            if exist:
                log.info("File Share '%s' on storage account '%s' created.", ref.remoteFileShare, ref.storageAccountName)
            else:
                log.info("File Share '%s' on storage account '%s' already existed.", ref.remoteFileShare, ref.storageAccountName)
            # Get most recent list of datastores linked to current workspace
            datastores = ws.datastores()
            # Validate if ds is created
            ds = None if ref.dataref_id not in datastores else Datastore(workspace = ws, name = ref.dataref_id)
            # Register the DS to the workspace
            if ds:
                if ds.account_name == ref.storageAccountName and ds.container_name == ref.remoteFileShare:
                    recreate = False
                else:
                    recreate = True
                    # also remove the existing reference
                    ds.unregister()
            else:
                recreate = True
            if recreate:
                log.info('Registering file share "{}" to AML datastore for AML workspace "{}" under datastore id "{}".'.format(ref.remoteFileShare, ws.name, ref.dataref_id))
                ds = Datastore.register_azure_file_share(workspace = ws,
                                                    datastore_name = ref.dataref_id, 
                                                    file_share_name = ref.remoteFileShare, 
                                                    account_name = ref.storageAccountName, 
                                                    account_key= ref.storageAccountKey,
                                                    overwrite=True,
                                                    )
            else:
                log.info('File share "{}" under AML workspace "{}" already registered under datastore id "{}".'.format(ref.remoteFileShare, ws.name, ref.dataref_id))

def upload_files_to_azure(cfg, ws):
    ''' look in the cfg object to file directories and files to upload to AFS and ABS
    input params :
    ws : Description : aml workspace object
    ws : Type : aml workspace object (defined in azureml.core.workspace.Workspace)
    '''
    for ref in cfg.DataReference.localDirectoryBlobList:
        uploadContentBeforeRun = ref.uploadContentBeforeRun
        if uploadContentBeforeRun:
            ds_ref_id = ref.dataref_id
            overwriteOnUpload = ref.overwriteOnUpload
            remoteBlobContainer = ref.remoteBlobContainer
            localDirectoryPath  = ref.localDirectoryPath
            remoteDirectoryPath = ref.remoteDirectoryPath
            ds = Datastore(workspace = ws, name = ds_ref_id)
            ds.upload(src_dir=localDirectoryPath, target_path=remoteDirectoryPath, overwrite=overwriteOnUpload, show_progress=True)

    for ref in cfg.DataReference.localDirectoryFilesList:
        uploadContentBeforeRun = ref.uploadContentBeforeRun
        if uploadContentBeforeRun:
            ds_ref_id = ref.dataref_id
            overwriteOnUpload = ref.overwriteOnUpload
            remoteFileShare = ref.remoteFileShare
            localDirectoryPath = ref.localDirectoryPath
            remoteDirectoryPath = ref.remoteDirectoryPath
            ds = Datastore(workspace = ws, name = ds_ref_id)
            ds.upload(src_dir = localDirectoryPath, target_path=remoteDirectoryPath, overwrite=overwriteOnUpload, show_progress=True)

def create_aml_workspace(cfg):
    """ Creates the AML workspace if it doesn't exist. If it does
    exist, return the existing one.
    input : cfg : AMLConfiguration object containing all creation parameters
    output : ws : type workspace
    """
    try:
        log.info('Trying to retrieve config file from local filesystem.')
        ws = Workspace.from_config()
        if ws.name == cfg.AMLConfig.workspace:
            log.info('Workspace found with name: ' + ws.name)
            log.info('  Azure region: ' + ws.location)
            log.info('  Subscription id: ' + ws.subscription_id)
            log.info('  Resource group: ' + ws.resource_group)
        else:
            log.error('Workspace found ({}), but not the same as in the JSON config file ({}). Please delete config folder (aml_config) and restart.'.format(ws.name, cfg.AMLConfig.workspace))
            exit(-2)
    except:
        log.info('Unable to find AML config files in (aml_config) - attempting to Creating them.')
        try:
            log.info('Creating the workspace on Azure.')
            ws = Workspace.create(name = cfg.AMLConfig.workspace, 
                auth = cfg.Credentials,
                subscription_id = cfg.subscription_id,
                resource_group = cfg.AMLConfig.resource_group, 
                location = cfg.AMLConfig.location,
                create_resource_group = True,
                exist_ok = False)
            log.info('Workspace created. Saving details to file in (aml_config) to accelerate further launches.')
            ws.get_details()
            ws.write_config()
        except Exception as exc:
            log.error('Unable to create the workspace on Azure. Error Message : ' + str(exc))
            exit(-2)
    return ws

def create_aml_compute_target_batchai(cfg, ws):
    """
    input : 
        ws : definition :  workspace 
                type : Workspace from azureml.core.workspace
        cfg : config dictionnary from the json file input for this program
                type : python dictionnary 
    output : computetarget object
    """
    try:
        compute_target = ComputeTarget(workspace = ws, name = cfg.ClusterProperties.cluster_name)
        log.info('Found existing compute target. Using it. NOT VALIDATING IF YOU CHANGED THE CLUSTER CONFIG...')
    except ComputeTargetException:
        log.info('Creating Batch AI compute target "{}" in workspace "{}".'.format(cfg.ClusterProperties.cluster_name, ws.name))
        # Defining the compute configuration for actual target creation
        compute_config = BatchAiCompute.provisioning_configuration(
                                            vm_size= cfg.ClusterProperties.vm_size,
                                            vm_priority= cfg.ClusterProperties.vm_priority,
                                            autoscale_enabled=True if cfg.ClusterProperties.scaling_method == 'auto_scale' else False,
                                            cluster_min_nodes=cfg.ClusterProperties.minimumNodeCount,
                                            cluster_max_nodes=cfg.ClusterProperties.maximumNodeCount,
                                            location = cfg.AMLConfig.location)
        log.info('Launching creation of the Batch AI compute target "{}" under the AML workspace "{}"'.format(cfg.ClusterProperties.cluster_name, ws.name))
        compute_target = ComputeTarget.create(workspace= ws, name=cfg.ClusterProperties.cluster_name, provisioning_configuration=compute_config)
        compute_target.wait_for_completion(show_output=True)
        log.info(compute_target.status.serialize())
    return compute_target

def create_aml_experiment(cfg, ws):
    """
    input : 
        cfg : config dictionnary from the json file input for this program
            type : python dictionnary 
        ws : definition :  workspace 
            type : Workspace from azureml.core.workspace
    output : Experiment from azureml.core.experiment
    """
    try:
        exp = Experiment(workspace = ws, name = cfg.AMLConfig.experimentation + "-" + cfg.JobProperties.jobNamePrefix + time.strftime("%Y%m%d-%H%M%S")) #lazy call - experiment created upon call f it
    except Exception as exc:
        log.error('Problem at Experiment object creation. Error = {}'.format(exc))
        exit(-2)
    return exp

def main(job_profile_file):
    """ Main file to run within AML or Batch AI
    input : job_profile_file : description : file containing the json parameters for the whole thing
            job_profile_path : type : string containing a path
            use_aml : true if using AML, false if using directly Batch AI
            use_aml : type : bool
    output : nothing
    """

    # Manually first cluster and job configuration
    cfg = AMLConfiguration(job_profile_file)

    # Create or retrieving the workspace (will create RG if required)
    ws = create_aml_workspace(cfg)

    # create blob container for dataset and file share for scripts & logs - see json for defaults
    create_and_attach_blob_storage(cfg, ws)
    create_and_attach_file_storage(cfg, ws)
    
    # Upload files to Azure (blob and files)
    upload_files_to_azure(cfg, ws)

    # Create the experimentation
    exp = create_aml_experiment(cfg, ws)

    # Create or acquire the compute target
    ct = create_aml_compute_target_batchai(cfg, ws)

    # Create the estimator (job prereq)
    estimator = cfg.JobProperties.jobEstimator.getAMLTensorFlowEstimator(ct, ws, cfg)

    # region - HyperDrive - Create the hyperrive param
    # ps = RandomParameterSampling(
    # ps = RandomParameterSampling(
    #     {
    #         '--batch':choice(8, 16, 32, 64),
    #         '--learning_rate':uniform(1.e-6, 1.e-2)
    #     }
    # )
    # policy = BanditPolicy(evaluation_interval=3, slack_factor=0.02, delay_evaluation=5)
    # htcestimator = HyperDriveRunConfig  (estimator = estimator,
    #                                     hyperparameter_sampling = ps,
    #                                     primary_metric_name = "Epoch Validation Loss", 
    #                                     primary_metric_goal = PrimaryMetricGoal.MINIMIZE,
    #                                     max_concurrent_runs = 4,
    #                                     max_total_runs = 20,
    #                                     #max_duration_minutes = 180,
    #                                     policy = policy)
    # htcrun = exp.submit(config = htcestimator)
    # htcrun.wait_for_completion(show_output=True)
    # # endregion

    # Start the job
    run = exp.submit(config = estimator)
    # Wait until the end and display the output
    run.wait_for_completion(show_output=True)

    # myrun = Run(experiment=exp.experiment, run_id = '..')
    # myrun.cancel()

    # TODO : write code to download back results (or which other dataref has been marked for it)

    return

if __name__ == "__main__":
    """ Parsing arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_profile_file', type=str,
                        help='JSON file containing all definitions (cluster, jobs, datasets, etc.)')
    inputArgs = parser.parse_args()
    
    main(job_profile_file = inputArgs.job_profile_file)

        




