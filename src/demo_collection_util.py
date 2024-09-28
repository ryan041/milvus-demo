from pymilvus import MilvusClient, DataType


class DemoCollectionMgr:
    def __init__(self, uri, db_name):
        self.uri = uri
        self.db_name = db_name
        self.client = MilvusClient(uri=self.uri, db_name=self.db_name)

    def create_deme_collection(self, name='demo'):
        # create a schema
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False, description='用于测试向量数据库功能')
        schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, description='数据ID值，自动生成')
        schema.add_field(field_name='vector', datatype=DataType.FLOAT_VECTOR, dim=1024, description='文本对应的向量值')
        schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=2048, description='文本')
        schema.add_field(field_name='subject', datatype=DataType.VARCHAR, max_length=64, description='文本主题')

        # create a index
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name='vector', index_type='IVF_FLAT', metric_type='IP', params={"nlist": 128})

        # create a collection
        self.client.create_collection(collection_name=name, schema=schema, index_params=index_params)

        # client.load_collection(collection_name=collectionName, replica_number=1)

    def get_client(self):
        return self.client




