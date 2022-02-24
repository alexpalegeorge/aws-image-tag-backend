import json
import boto3

had_no_objects = False

def lambda_handler(event, context):

    url = event['body']['url']
    input_tags = event['body']['tags']
    
    image_name = url.split('/')[-1]
    
    
    # get item from dynamodb
    dynamodb = boto3.client('dynamodb')
    item = dynamodb.query(
        TableName='tagtag-sorted',
        IndexName='image-index',
        KeyConditionExpression= 'image = :pk',
        ExpressionAttributeValues= {
            ':pk': {
                'S': image_name
            }
        },
        ProjectionExpression= 'image, tag'
    )
    
    # make sure image at given url exists, then add new tags
    if "Items" in item:
        new_tags = make_new_tags(item, input_tags)
        for t in input_tags:
            add_to_db(image_name, t)
            
        if had_no_objects:
            delete_no_objects_entry(image_name)
            
        response = {
        'statusCode': 200,
        'url': url,
        'updated data': new_tags
    }
    else:
        response = {
            'statusCode': 200,
            'url': 'does not exist'
        }
    
    return response
    

def make_new_tags(item, input_tags):
    current_tags = []
    for i in item['Items']:
        this_tag = i['tag']['S']
        if this_tag == 'no objects':
            had_no_objects = True
        current_tags.append(this_tag)
        
    
    for t in input_tags:
        if t not in current_tags:
            current_tags.append(t)
    return current_tags

def add_to_db(image_name, add_tag):
    
    dynamodb = boto3.resource('dynamodb')
    dynamo_table = dynamodb.Table('tagtag-sorted')
    
    dynamo_table.put_item(
        Item = {
            'tag': add_tag,
            'image': image_name
        }
    )
    
def delete_no_objects_entry(image_name):
    
    dynamodb = boto3.resource('dynamodb')
    dynamo_table = dynamodb.Table('tagtag-sorted')
    
    dynamo_table.delete_item(
        Item = {
            'tag': 'no objects',
            'image': image_name
        }
    )
