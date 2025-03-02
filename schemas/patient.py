from pymongo.encryption import Algorithm, ClientEncryption

# Define the JSON schema for the patient collection
patient_schema = {
    'bsonType': 'object',
    'properties': {
        'patient_id': {'bsonType': 'string'},
        'name': {'bsonType': 'string'},
        'nic': {'bsonType': 'string', 'encrypt': {'bsonType': 'string'}},
        'dob': {'bsonType': 'string', 'encrypt': {'bsonType': 'string'}},
        'vaccine': {'bsonType': 'string'},
        'address': {'bsonType': 'string', 'encrypt': {'bsonType': 'string'}},
        'gender': {'bsonType': 'string'},
        'age': {'bsonType': 'string'},
        'weight': {'bsonType': 'string'},
        'height': {'bsonType': 'string'},
        'email': {'bsonType': 'string', 'encrypt': {'bsonType': 'string'}},
        'phone_number': {'bsonType': 'string', 'encrypt': {'bsonType': 'string'}},
        'insurance_date': {'bsonType': 'string'},
        'file': {'bsonType': 'string'}
    },
    'required': ['patient_id', 'name', 'nic', 'dob', 'address', 'email', 'phone_number']
}

# Define the JSON schema for the data key
key_schema = {
    'bsonType': 'object',
    'properties': {
        'keyMaterial': {'bsonType': 'binary', 'description': 'binary representation of the key'},
        'creationDate': {'bsonType': 'date', 'description': 'key creation timestamp'},
        'updatedDate': {'bsonType': 'date', 'description': 'key last modified timestamp'},
        'status': {'enum': ['ACTIVE', 'DEPRECATED', 'INVALID'], 'description': 'key status'},
        'masterKey': {'bsonType': 'objectId', 'description': 'ID of the master key used to encrypt the key material'},
        'keyAltNames': {'bsonType': 'array', 'items': {'bsonType': 'string'}}
    },
    'required': ['keyMaterial', 'masterKey']
}

# Create a ClientEncryption instance for encrypting and decrypting data
client_encryption = ClientEncryption(client, {
    'keyVaultNamespace': 'encryption.__keyVault',
    'kmsProviders': {
        'local': {
            'key': master_key
        }
    }
})

# Define the encryption options for the patient collection
patient_collection_options = {
    'validator': {'$jsonSchema': patient_schema},
    'encryption': {
        'keyId': client_encryption.createDataKey('local', key_schema),
        'algorithm': Algorithm.AEAD_AES_256_CBC_HMAC_SHA_512_Deterministic,
        'bsonType': 'object',
        'extraOptions': {
            'mongocryptdSpawnArgs': ['--idleShutdownTimeoutSecs=60']
        }
    }
}

# Create the patient collection with encryption options
patient_collection = db.create_collection(
    'patients', **patient_collection_options)
