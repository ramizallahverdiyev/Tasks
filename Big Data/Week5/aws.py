import boto3

ec2 = boto3.resource('ec2',
                     region_name='us-east-1',
                     aws_access_key_id='YOUR_ACCESS_KEY',
                     aws_secret_access_key='YOUR_SECRET_KEY')

instances = ec2.create_instances(
    ImageId='ami-0c02fb55956c7d316',
    MinCount=1,
    MaxCount=1,
    InstanceType='t2.micro',
    KeyName='your-key-pair-name'      
)

instance_id = instances[0].id
print(f"EC2 instance yaradıldı: {instance_id}")

ec2_client = boto3.client('ec2',
                          region_name='us-east-1',
                          aws_access_key_id='YOUR_ACCESS_KEY',
                          aws_secret_access_key='YOUR_SECRET_KEY')

ec2_client.terminate_instances(InstanceIds=[instance_id])
print(f"EC2 instance silindi: {instance_id}")
