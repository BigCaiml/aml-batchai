from __future__ import print_function

import json
import os

from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.train.dnn import TensorFlow

class AMLConfiguration:
    """Configuration for recipes and notebooks"""

    def __init__(self, file_name):
        if not os.path.exists(file_name):
            raise ValueError('Cannot find configuration file "{0}"'.
                             format(file_name))

        with open(file_name, 'r') as f:
            conf = json.load(f)

        try:
            # region AMLConfig Section
            amlconf = conf['AMLConfig']
            resource_group = self._encode(amlconf['resource_group'])
            location = self._encode(amlconf['location'])
            workspace = self._encode(amlconf['workspace'])
            experimentation = self._encode(amlconf['experimentation'])
            self.AMLConfig = self.__AMLConfig(resource_group = resource_group, 
                                                location = location, 
                                                workspace = workspace, 
                                                experimentation = experimentation)
            # endregion
            
            # region Credentials Section
            creds = conf['Credentials']
            self.cred_type = creds['type']
            self.subscription_id = creds['subscription_id']
            if self.cred_type == 'sp_credentials' and 'sp_credentials' in creds:
                spcreds = creds['sp_credentials']
                aad_client_id = spcreds['aad_client_id']
                aad_secret_key = spcreds['aad_secret']
                aad_tenant = spcreds['aad_tenant']
                self.Credentials = ServicePrincipalAuthentication(tenant_id = aad_tenant,
                                                                username = aad_client_id,
                                                                password = aad_secret_key)
            elif self.cred_type == 'userpass_credentials':
                self.Credentials = InteractiveLoginAuthentication(force=False)
            # endregion

            # region ClusterProperties Section
            clusterProperties = conf['ClusterProperties']
            vmPriority = clusterProperties['vm_priority']
            vmSize = clusterProperties['vm_size']
            cluster_name = self._encode(clusterProperties['cluster_name'])
            scaling_method = clusterProperties['scaling']['scaling_method']
            scaling = clusterProperties['scaling'][scaling_method]
            if scaling_method=="manual":
                minimumNodeCount = scaling['target_node_count']
                maximumNodeCount = minimumNodeCount
            elif scaling_method=="auto_scale":
                minimumNodeCount = scaling['minimum_node_count']
                maximumNodeCount = scaling['maximum_node_count']
            else:
                raise ("Parsing error, scaling undefined - needs to be manual or auto_scale")
            self.ClusterProperties = self.__AMLClusterProperties(vm_size = vmSize,
                                                                vm_priority = vmPriority,
                                                                scaling_method = scaling_method,
                                                                minimumNodeCount = minimumNodeCount,
                                                                maximumNodeCount = maximumNodeCount,
                                                                cluster_name = cluster_name)

            # endregion

            # region JobProperties Section            
            jobProperties = conf['JobProperties']
            jobNamePrefix = str(jobProperties['jobNamePrefix'])
            jobNodeCount = int(jobProperties['nodeCount'])
            jobProcessCount = int(jobProperties['processCount'])
            jobEstimator = jobProperties['estimator']
            jobEstimatorType = jobEstimator['estimatorType']
            jobScript = jobEstimator['script']
            jobScriptPath = jobEstimator['scriptPath']
            jobScriptArgs = jobEstimator['scriptArgsDict']
            jobDistributedBackEnd = jobEstimator['distributedBackEnd']
            jobPipPackages = jobEstimator['pipPackages']
            # Create the estimator based on the type (they might be tensorflow, pytorch or base)
            self.JobProperties = self.__AMLJobProperties(
                jobNamePrefix = jobNamePrefix,
                jobEstimatorType = jobEstimatorType,
                jobNodeCount = jobNodeCount,
                jobProcessCount = jobProcessCount,
                jobScriptPath = jobScriptPath,
                jobScript = jobScript,
                jobScriptArgs = jobScriptArgs,
                jobDistributedBackEnd = jobDistributedBackEnd,
                jobPipPackages = jobPipPackages
            )
            # endregion

            # region DataReference Section
            dataReference = conf['DataReferences']
            # Loop through list of Files Directories
            localDirectoryBlobList = []
            try:
                for ref in dataReference['localDirectoryBlob']:
                    localDirectoryBlobList.append(self.__AMLBlobDataRef(
                        dataref_id = ref['dataref_id'],
                        localDirectoryName = ref['localDirectoryName'],
                        remoteMountPath = ref['remoteMountPath'],
                        downloadToComputeNodeBeforeExecution = ref['downloadToComputeNodeBeforeExecution'].upper() == "TRUE",
                        remoteBlobContainer = ref['remoteBlobContainer'],
                        uploadContentBeforeRun = ref['uploadContentBeforeRun'].upper() == "TRUE",
                        overwriteOnUpload = ref['overwriteOnUpload'].upper() == "TRUE",
                        downloadContentAfterRun = ref['downloadContentAfterRun'].upper() == "TRUE",
                        storageAccountName = ref['storageAccountName'],
                        storageAccountKey = ref['storageAccountKey'])
                    )
            except KeyError as err:
                # Key not present in json config
                pass
            localDirectoryFilesList = []
            try:
                for ref in dataReference['localDirectoryFiles']:
                    localDirectoryFilesList.append(self.__AMLFilesDataRef(
                        dataref_id = ref['dataref_id'],
                        localDirectoryName = ref['localDirectoryName'],
                        remoteMountPath = ref['remoteMountPath'],
                        downloadToComputeNodeBeforeExecution = ref['downloadToComputeNodeBeforeExecution'].upper() == "TRUE",
                        remoteFileShare = ref['remoteFileShare'],
                        uploadContentBeforeRun = ref['uploadContentBeforeRun'].upper() == "TRUE",
                        overwriteOnUpload = ref['overwriteOnUpload'].upper() == "TRUE",
                        downloadContentAfterRun = ref['downloadContentAfterRun'].upper() == "TRUE",
                        storageAccountName = ref['storageAccountName'],
                        storageAccountKey = ref['storageAccountKey'])
                    )
            except KeyError as err:
                # Key not present in json config
                pass
            self.DataReference = self.__DataReference(
                localDirectoryBlobList = localDirectoryBlobList,
                localDirectoryFilesList = localDirectoryFilesList
            )
            # endregion

        except KeyError as err:
            raise AttributeError('Please provide a value for "{0}" configuration key'.format(err.args[0]))
        except Exception as err:
            raise ('Error in config parsing : ' + str(err))

    @staticmethod
    def _encode(value):
        if (isinstance(value, type('str')) or isinstance(value, int)):
            return value
        return value.encode('utf-8')

    class __AMLConfig(object):
        """ AML Config section of the json document
        """
        def __init__(self, **kwargs):
            self.resource_group = kwargs.get('resource_group',None)
            self.location = kwargs.get('location', None)
            self.workspace = kwargs.get('workspace', None)
            self.experimentation = kwargs.get('experimentation', None)

    class __AMLClusterProperties(object):
        ''' description
            'vm_size': {'key': 'vm_size', 'type': 'str'},
            'vm_priority': {'key': 'vm_priority', 'type': 'string'},
            'scaling_method': {'key': 'scaling_method', 'type' : 'string'},
            'cluster_name': {'key': 'cluster_name', 'type': 'str'},
        '''
        def __init__(self, **kwargs):
            self.vm_size = kwargs.get('vm_size', None)
            self.vm_priority = kwargs.get('vm_priority', None)
            self.scaling_method = kwargs.get('scaling_method', None)
            self.minimumNodeCount = kwargs.get('minimumNodeCount', None)
            self.maximumNodeCount = kwargs.get('maximumNodeCount', None) 
            self.cluster_name = kwargs.get('cluster_name', None)
    
    class __AMLJobProperties(object):
        ''' Description
            'containerSettings': {'key': 'containerSettings', 'type': 'ContainerSettings'},
            'stdOutErrPathPrefix': {'key': 'stdOutErrPathPrefix', 'type': 'str'},
            'jobNamePrefix': {'key': 'jobNamePrefix', 'type': 'str'},
            'mountVolumes': {'key': 'mountVolumes', 'type': 'models.MountVolumes'},
            'tensorFlowSettings': {'key': 'TensorFlowSettings', 'type': 'models.TensorFlowSettings'},
            'horovodSettings': {'key': 'HorovodSettings', 'type': 'models.HorovodSettings'},
            'localDatasetDirectoryName': {'key': 'localDatasetDirectoryName', 'type': 'str'},
            'localTrainingScriptFile': {'key': 'localTrainingScriptFile', 'type': 'str'},
            'jobPrepCommandLine': {'key': 'jobPrepCommandLine', 'type': 'models.JobPreparation'},
            'nodeCount': {'key': 'nodeCount', 'type': 'int'},
            stdOutErrMonitoredFile : file name to check and tail output until done.
        '''

        def __init__(self, **kwargs):
            self.jobNamePrefix = kwargs.get('jobNamePrefix', None)
            self.jobEstimatorType = kwargs.get('jobEstimatorType', None)
            jobNodeCount = kwargs.get('jobNodeCount', None)
            jobProcessCount = kwargs.get('jobProcessCount', None)
            jobScriptPath = kwargs.get('jobScriptPath', None)
            jobScript = kwargs.get('jobScript', None)
            jobScriptArgs = kwargs.get('jobScriptArgs', None)
            jobDistributedBackEnd = kwargs.get('jobDistributedBackEnd', None)
            jobPipPackages = kwargs.get('jobPipPackages', None)

            jobProcessCountPerNode = int(jobProcessCount/jobNodeCount)
            dataReference = kwargs.get('dataReference', None)

            if self.jobEstimatorType == "tensorflow":
                self.jobEstimator = self.____AMLEstimatorProperties(
                    source_directory = jobScriptPath,
                    entry_script = jobScript,
                    script_params = jobScriptArgs,
                    node_count = jobNodeCount,
                    process_count_per_node = jobProcessCountPerNode,
                    distributed_backend = jobDistributedBackEnd,
                    use_gpu = True,
                    use_docker = True,
                    pip_packages = jobPipPackages)

        class ____AMLEstimatorProperties(object):
            ''' Description : internal class to prepare estimator objects for AML
            '''
            def __init__(self, **kwargs):
                self._source_directory          = kwargs.get('source_directory', None)
                self._entry_script              = kwargs.get('entry_script', None)
                self._script_params             = kwargs.get('script_params', None)
                self._node_count                = int(kwargs.get('node_count', 1))
                self._process_count_per_node    = int(kwargs.get('process_count_per_node', 1))
                self._distributed_backend       = kwargs.get('distributed_backend', 'mpi')
                self._use_gpu                   = kwargs.get('use_gpu', None)
                self._use_docker                = kwargs.get('use_docker', True)
                self._custom_docker_base_image  = kwargs.get('custom_docker_base_image', None)
                self._pip_packages              = kwargs.get('pip_packages', None)
                self._environment_definition    = kwargs.get('environment_definition', None)
                self._inputs                    = kwargs.get('inputs',None)

            def getAMLTensorFlowEstimator(self, compute_target, ws, cfg):
                self._compute_target = compute_target
                self._doctor_script_parameters_for_datastore(ws, cfg)
                return TensorFlow(
                    source_directory = self._source_directory,
                    compute_target = self._compute_target,
                    entry_script = self._entry_script,
                    script_params = self._script_params,
                    node_count = self._node_count,
                    process_count_per_node = self._process_count_per_node,
                    distributed_backend = self._distributed_backend,
                    use_gpu = self._use_gpu,
                    use_docker = self._use_docker,
                    pip_packages = self._pip_packages,
                    environment_definition = self._environment_definition,
                    inputs = self._inputs)

            def _doctor_script_parameters_for_datastore(self, ws, cfg):
                # Find parameters to be doctored
                # Create a k-v list of datareferences in the cfg object
                datarefs = {}
                for blobref in cfg.DataReference.localDirectoryBlobList:
                    datarefs[blobref.dataref_id] = blobref.downloadToComputeNodeBeforeExecution
                for fileref in cfg.DataReference.localDirectoryFilesList:
                    datarefs[fileref.dataref_id] = fileref.downloadToComputeNodeBeforeExecution

                # Get a reference to the datastores for this particular workspace
                datastores = ws.datastores()
                # Go through tht list of parameters in the json config file so that we replace the %%tfmodel 
                # with the proper environment variable (like AZBATCHAI_INPUT_...)
                for key, value in self._script_params.items():
                    # Look if the input param starts with %%
                    if value[0:2] == '%%': 
                        # Find the datastore name related (removing the initial %% in the parameter if there is one)
                        dataref_id = value.split('/')[0][2:]
                        relative_path_on_compute = self._script_params[key][2+len(dataref_id)+1:] # Find the extra (after the / if there is one )
                        ds = datastores[dataref_id]
                        dr = ds.as_download() if datarefs[dataref_id] else ds.as_mount() #as_mount() if self._ ._data_references['downloadToComputeNodeBeforeExecution'] = True else ds.as_download()
                        dr.path_on_datastore = relative_path_on_compute if len(relative_path_on_compute) > 0 else None
                        self._script_params[key] = dr 

    class __AMLBlobDataRef(object):
        def __init__(self, **kwargs):
            self.dataref_id = kwargs.get('dataref_id',None)
            self.localDirectoryName = kwargs.get('localDirectoryName', None)
            self.remoteMountPath = kwargs.get('remoteMountPath', None)
            self.downloadToComputeNodeBeforeExecution = kwargs.get('downloadToComputeNodeBeforeExecution', None)
            self.remoteBlobContainer = kwargs.get('remoteBlobContainer', None)
            self.uploadContentBeforeRun = kwargs.get('uploadContentBeforeRun', False)
            self.overwriteOnUpload = kwargs.get('overwriteOnUpload', False)
            self.downloadContentAfterRun = kwargs.get('downloadContentAfterRun', False)
            self.storageAccountName = kwargs.get('storageAccountName', None)
            self.storageAccountKey = kwargs.get('storageAccountKey', None)

    class __AMLFilesDataRef(object):
        def __init__(self, **kwargs):
            self.dataref_id = kwargs.get('dataref_id',None)
            self.localDirectoryName = kwargs.get('localDirectoryName', None)
            self.remoteMountPath = kwargs.get('remoteMountPath', None)
            self.downloadToComputeNodeBeforeExecution = kwargs.get('downloadToComputeNodeBeforeExecution', None)
            self.remoteFileShare = kwargs.get('remoteFileShare', None)
            self.uploadContentBeforeRun = kwargs.get('uploadContentBeforeRun', False)
            self.overwriteOnUpload = kwargs.get('overwriteOnUpload', False)
            self.downloadContentAfterRun = kwargs.get('downloadContentAfterRun', False)
            self.storageAccountName = kwargs.get('storageAccountName', None)
            self.storageAccountKey = kwargs.get('storageAccountKey', None)

    class __DataReference(object):
        def __init__(self, **kwargs):
            self.localDirectoryBlobList = kwargs.get('localDirectoryBlobList',None)
            self.localDirectoryFilesList = kwargs.get('localDirectoryFilesList', None)

