#!/usr/bin/env python3
"""
EKS CloudWatch Data Collector

Collects performance metrics and pricing data for EKS clusters and related resources.
Outputs data in CSV format for AI agent consumption.

Usage:
    python eks_cloudwatch_collector.py --cluster-name my-cluster --profile my-aws-profile
"""

import argparse
import boto3
import csv
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AWS Pricing per hour (us-east-1 prices as baseline, adjust as needed)
PRICING = {
    # EC2 Instance pricing per hour (sample prices - update with current pricing)
    'ec2': {
        't3.micro': 0.0104,
        't3.small': 0.0208,
        't3.medium': 0.0416,
        't3.large': 0.0832,
        't3.xlarge': 0.1664,
        't3.2xlarge': 0.3328,
        'm5.large': 0.096,
        'm5.xlarge': 0.192,
        'm5.2xlarge': 0.384,
        'm5.4xlarge': 0.768,
        'm5.8xlarge': 1.536,
        'm5.12xlarge': 2.304,
        'm5.16xlarge': 3.072,
        'm5.24xlarge': 4.608,
        'c5.large': 0.085,
        'c5.xlarge': 0.17,
        'c5.2xlarge': 0.34,
        'c5.4xlarge': 0.68,
        # R6g instances (ARM Graviton2 - eu-west-1 pricing)
        'r6g.large': 0.0864,
        'r6g.xlarge': 0.1728,
        'r6g.2xlarge': 0.3456,
        'r6g.4xlarge': 0.6912,
        'r6g.8xlarge': 1.3824,
        'r6g.12xlarge': 2.0736,
        'r6g.16xlarge': 2.7648,
        # R6a instances (AMD - eu-west-1 pricing)  
        'r6a.large': 0.0864,
        'r6a.xlarge': 0.1728,
        'r6a.2xlarge': 0.3456,
        'r6a.4xlarge': 0.6912,
        'r6a.8xlarge': 1.3824,
        'r6a.12xlarge': 2.0736,
        'r6a.16xlarge': 2.7648,
        'c5.9xlarge': 1.53,
        'c5.12xlarge': 2.04,
        'c5.18xlarge': 3.06,
        'c5.24xlarge': 4.08,
        'r5.large': 0.126,
        'r5.xlarge': 0.252,
        'r5.2xlarge': 0.504,
        'r5.4xlarge': 1.008,
        'r5.8xlarge': 2.016,
        'r5.12xlarge': 3.024,
        'r5.16xlarge': 4.032,
        'r5.24xlarge': 6.048,
    },
     # EBS pricing per GB per hour
     'ebs': {
         'gp2': 0.10 / 24 / 30,  # $0.10 per GB per month
         'gp3': 0.08 / 24 / 30,  # $0.08 per GB per month
         'io1': 0.125 / 24 / 30, # $0.125 per GB per month
         'io2': 0.125 / 24 / 30, # $0.125 per GB per month
         'st1': 0.045 / 24 / 30, # $0.045 per GB per month
         'sc1': 0.025 / 24 / 30, # $0.025 per GB per month
     },
     # EBS Snapshot pricing per GB per hour (eu-west-1 pricing)
     'ebs_snapshot': 0.05 / 24 / 30,  # $0.05 per GB per month
    # EKS cluster pricing per hour
    'eks_cluster': 0.6,   # $0.60 per hour per cluster
    # Load balancer pricing per hour
    'alb': 0.0225,  # $0.0225 per hour
    'nlb': 0.0225,  # $0.0225 per hour
    'clb': 0.025,   # $0.025 per hour
}

class EKSCloudWatchCollector:
    def __init__(self, profile_name: str, region: str = 'us-east-1'):
        """Initialize AWS clients with specified profile."""
        self.session = boto3.Session(profile_name=profile_name, region_name=region)
        self.eks_client = self.session.client('eks')
        self.ec2_client = self.session.client('ec2')
        self.cloudwatch = self.session.client('cloudwatch')
        self.elbv2_client = self.session.client('elbv2')
        self.elb_client = self.session.client('elb')
        self.region = region
        
    def get_cluster_resources(self, cluster_name: str) -> Dict[str, List[Dict]]:
        """Discover all resources related to the EKS cluster."""
        logger.info(f"Discovering resources for EKS cluster: {cluster_name}")
        
        resources = {
            'cluster': [],
            'instances': [],
            'volumes': [],
            'snapshots': [],
            'load_balancers': []
        }
        
        try:
            # Get cluster info
            cluster_info = self.eks_client.describe_cluster(name=cluster_name)
            resources['cluster'].append({
                'name': cluster_name,
                'status': cluster_info['cluster']['status'],
                'version': cluster_info['cluster']['version'],
                'created_at': cluster_info['cluster']['createdAt']
            })
            
            # Get node groups
            nodegroups_response = self.eks_client.list_nodegroups(clusterName=cluster_name)
            
            instance_ids = []
            for ng_name in nodegroups_response['nodegroups']:
                ng_info = self.eks_client.describe_nodegroup(
                    clusterName=cluster_name,
                    nodegroupName=ng_name
                )
                
                # Get ASG instances
                if 'resources' in ng_info['nodegroup'] and 'autoScalingGroups' in ng_info['nodegroup']['resources']:
                    for asg in ng_info['nodegroup']['resources']['autoScalingGroups']:
                        asg_name = asg['name']
                        # Get instances from ASG
                        asg_instances = self._get_asg_instances(asg_name)
                        instance_ids.extend(asg_instances)
            
            # Get instance details
            if instance_ids:
                instances_response = self.ec2_client.describe_instances(InstanceIds=instance_ids)
                for reservation in instances_response['Reservations']:
                    for instance in reservation['Instances']:
                        resources['instances'].append({
                            'instance_id': instance['InstanceId'],
                            'instance_type': instance['InstanceType'],
                            'state': instance['State']['Name'],
                            'launch_time': instance['LaunchTime'],
                            'availability_zone': instance['Placement']['AvailabilityZone']
                        })
                        
                        # Get attached volumes
                        for bdm in instance.get('BlockDeviceMappings', []):
                            if 'Ebs' in bdm:
                                volume_id = bdm['Ebs']['VolumeId']
                                volume_info = self._get_volume_info(volume_id)
                                if volume_info:
                                    resources['volumes'].append(volume_info)
            
            # Get snapshots for discovered volumes
            resources['snapshots'] = self._get_volume_snapshots([vol['volume_id'] for vol in resources['volumes']])
            
            # Get load balancers (this is more complex, we'll look for LBs with EKS tags)
            resources['load_balancers'] = self._get_cluster_load_balancers(cluster_name)
            
            logger.info(f"Found {len(resources['instances'])} instances, {len(resources['volumes'])} volumes, {len(resources['snapshots'])} snapshots, {len(resources['load_balancers'])} load balancers")
            
        except Exception as e:
            logger.error(f"Error discovering cluster resources: {str(e)}")
            raise
            
        return resources
    
    def _get_asg_instances(self, asg_name: str) -> List[str]:
        """Get instance IDs from Auto Scaling Group."""
        try:
            autoscaling = self.session.client('autoscaling')
            response = autoscaling.describe_auto_scaling_groups(
                AutoScalingGroupNames=[asg_name]
            )
            
            instance_ids = []
            for asg in response['AutoScalingGroups']:
                for instance in asg['Instances']:
                    if instance['LifecycleState'] == 'InService':
                        instance_ids.append(instance['InstanceId'])
            
            return instance_ids
        except Exception as e:
            logger.warning(f"Error getting ASG instances for {asg_name}: {str(e)}")
            return []
    
    def _get_volume_info(self, volume_id: str) -> Optional[Dict]:
        """Get EBS volume information."""
        try:
            response = self.ec2_client.describe_volumes(VolumeIds=[volume_id])
            if response['Volumes']:
                volume = response['Volumes'][0]
                return {
                    'volume_id': volume_id,
                    'volume_type': volume['VolumeType'],
                    'size_gb': volume['Size'],
                    'state': volume['State'],
                    'created_time': volume['CreateTime']
                }
        except Exception as e:
            logger.warning(f"Error getting volume info for {volume_id}: {str(e)}")
            return None
    
    def _get_volume_snapshots(self, volume_ids: List[str]) -> List[Dict]:
        """Get EBS snapshots for the given volume IDs."""
        if not volume_ids:
            return []
            
        try:
            # Get snapshots for all volumes
            response = self.ec2_client.describe_snapshots(
                OwnerIds=['self'],  # Only snapshots owned by this account
                Filters=[
                    {
                        'Name': 'volume-id',
                        'Values': volume_ids
                    },
                    {
                        'Name': 'status',
                        'Values': ['completed']  # Only completed snapshots
                    }
                ]
            )
            
            snapshots = []
            for snapshot in response['Snapshots']:
                snapshots.append({
                    'snapshot_id': snapshot['SnapshotId'],
                    'volume_id': snapshot['VolumeId'],
                    'size_gb': snapshot['VolumeSize'],
                    'start_time': snapshot['StartTime'],
                    'state': snapshot['State'],
                    'description': snapshot.get('Description', ''),
                    'encrypted': snapshot.get('Encrypted', False)
                })
            
            return snapshots
            
        except Exception as e:
            logger.warning(f"Error getting snapshots: {str(e)}")
            return []
    
    def _get_cluster_load_balancers(self, cluster_name: str) -> List[Dict]:
        """Find load balancers associated with the EKS cluster."""
        load_balancers = []
        
        try:
            # Get ALBs/NLBs
            elbv2_response = self.elbv2_client.describe_load_balancers()
            for lb in elbv2_response['LoadBalancers']:
                # Check tags to see if it's associated with our cluster
                tags_response = self.elbv2_client.describe_tags(
                    ResourceArns=[lb['LoadBalancerArn']]
                )
                
                for tag_desc in tags_response['TagDescriptions']:
                    for tag in tag_desc['Tags']:
                        if (tag['Key'] == 'kubernetes.io/cluster/' + cluster_name or 
                            tag['Key'] == 'elbv2.k8s.aws/cluster' and cluster_name in tag['Value']):
                            load_balancers.append({
                                'name': lb['LoadBalancerName'],
                                'type': lb['Type'].lower(),
                                'scheme': lb['Scheme'],
                                'state': lb['State']['Code'],
                                'created_time': lb['CreatedTime']
                            })
                            break
            
            # Get Classic Load Balancers
            elb_response = self.elb_client.describe_load_balancers()
            for lb in elb_response['LoadBalancerDescriptions']:
                tags_response = self.elb_client.describe_tags(
                    LoadBalancerNames=[lb['LoadBalancerName']]
                )
                
                for tag_desc in tags_response['TagDescriptions']:
                    for tag in tag_desc['Tags']:
                        if tag['Key'] == 'kubernetes.io/cluster/' + cluster_name:
                            load_balancers.append({
                                'name': lb['LoadBalancerName'],
                                'type': 'clb',
                                'scheme': lb['Scheme'],
                                'state': 'active',
                                'created_time': lb['CreatedTime']
                            })
                            break
                            
        except Exception as e:
            logger.warning(f"Error getting load balancers: {str(e)}")
        
        return load_balancers
    
    def get_cloudwatch_metrics(self, resources: Dict, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Collect CloudWatch metrics for all resources."""
        logger.info(f"Collecting metrics from {start_time} to {end_time}")
        
        metrics_data = []
        
        # EKS Cluster metrics (limited metrics available)
        for cluster in resources['cluster']:
            cluster_metrics = self._get_cluster_metrics(cluster['name'], start_time, end_time)
            metrics_data.extend(cluster_metrics)
        
        # EC2 Instance metrics
        for instance in resources['instances']:
            instance_metrics = self._get_instance_metrics(instance, start_time, end_time)
            metrics_data.extend(instance_metrics)
        
        # EBS Volume metrics
        for volume in resources['volumes']:
            volume_metrics = self._get_volume_metrics(volume, start_time, end_time)
            metrics_data.extend(volume_metrics)
        
        # EBS Snapshot metrics (cost only, no CloudWatch metrics)
        for snapshot in resources['snapshots']:
            snapshot_metrics = self._get_snapshot_metrics(snapshot, start_time, end_time)
            metrics_data.extend(snapshot_metrics)
        
        # Load Balancer metrics
        for lb in resources['load_balancers']:
            lb_metrics = self._get_load_balancer_metrics(lb, start_time, end_time)
            metrics_data.extend(lb_metrics)
        
        logger.info(f"Collected {len(metrics_data)} metric data points")
        return metrics_data
    
    def _get_cluster_pricing_by_version(self, version: str) -> float:
        """Get EKS cluster pricing based on version."""
        # Parse version number (e.g., "1.28" from "1.28.5")
        try:
            major_minor = '.'.join(version.split('.')[:2])
            version_float = float(major_minor)
        except (ValueError, IndexError):
            # Default to base price if version parsing fails
            return PRICING['eks_cluster']
        
        # Version-based pricing logic
        if version_float >= 1.30:
            return 0.1  # $0.10/hour for version 1.30+
        else:
            return 0.6  # $0.60/hour for versions < 1.30

    def _get_cluster_metrics(self, cluster_name: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get EKS cluster metrics with version-based pricing."""
        metrics = []
        
        # Get cluster version for pricing
        try:
            cluster_info = self.eks_client.describe_cluster(name=cluster_name)
            cluster_version = cluster_info['cluster']['version']
            cluster_price_per_hour = self._get_cluster_pricing_by_version(cluster_version)
        except Exception as e:
            logger.warning(f"Could not get cluster version for {cluster_name}: {str(e)}")
            cluster_version = "unknown"
            cluster_price_per_hour = PRICING['eks_cluster']
        
        # Generate cost data points for each day
        current_time = start_time
        while current_time <= end_time:
            metrics.append({
                'timestamp': current_time.isoformat(),
                'resource_type': 'eks_cluster',
                'resource_id': cluster_name,
                'metric_name': 'cluster_cost',
                'value': cluster_price_per_hour * 24,  # Daily cost (24 hours)
                'unit': 'USD/day',
                'price_per_day': cluster_price_per_hour * 24,
                'cluster_version': cluster_version
            })
            current_time += timedelta(days=1)
        
        return metrics
    
    def _get_instance_metrics(self, instance: Dict, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get EC2 instance CloudWatch metrics."""
        metrics = []
        instance_id = instance['instance_id']
        instance_type = instance['instance_type']
        
        # Get pricing for this instance type (convert to daily)
        price_per_day = PRICING['ec2'].get(instance_type, 0.0) * 24
        
        metric_queries = [
            ('CPUUtilization', 'AWS/EC2', 'Percent'),
            ('DiskReadBytes', 'AWS/EC2', 'Bytes'),
            ('DiskWriteBytes', 'AWS/EC2', 'Bytes'),
            ('DiskReadOps', 'AWS/EC2', 'Count'),
            ('DiskWriteOps', 'AWS/EC2', 'Count'),
        ]
        
        for metric_name, namespace, unit in metric_queries:
            try:
                response = self.cloudwatch.get_metric_statistics(
                    Namespace=namespace,
                    MetricName=metric_name,
                    Dimensions=[
                        {
                            'Name': 'InstanceId',
                            'Value': instance_id
                        }
                    ],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=86400,  # 1 day
                    Statistics=['Average']
                )
                
                for datapoint in response['Datapoints']:
                    metrics.append({
                        'timestamp': datapoint['Timestamp'].isoformat(),
                        'resource_type': 'ec2_instance',
                        'resource_id': instance_id,
                        'instance_type': instance_type,
                        'metric_name': metric_name,
                        'value': datapoint['Average'],
                        'unit': unit,
                        'price_per_day': price_per_day
                    })
                    
            except Exception as e:
                logger.warning(f"Error getting {metric_name} for {instance_id}: {str(e)}")
        
        return metrics
    
    def _get_volume_metrics(self, volume: Dict, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get EBS volume CloudWatch metrics."""
        metrics = []
        volume_id = volume['volume_id']
        volume_type = volume['volume_type']
        size_gb = volume['size_gb']
        
        # Calculate pricing per day
        price_per_gb_hour = PRICING['ebs'].get(volume_type, PRICING['ebs']['gp2'])
        price_per_day = price_per_gb_hour * size_gb * 24
        
        # Simplified: Just track volume cost over time (no I/O metrics)
        current_time = start_time
        while current_time <= end_time:
            metrics.append({
                'timestamp': current_time.isoformat(),
                'resource_type': 'ebs_volume',
                'resource_id': volume_id,
                'volume_type': volume_type,
                'size_gb': size_gb,
                'metric_name': 'volume_cost',
                'value': price_per_day,  # Daily cost
                'unit': 'USD/day',
                'price_per_day': price_per_day
            })
            current_time += timedelta(days=1)
        
        return metrics
    
    def _get_snapshot_metrics(self, snapshot: Dict, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get EBS snapshot cost metrics (no CloudWatch metrics available for snapshots)."""
        metrics = []
        snapshot_id = snapshot['snapshot_id']
        volume_id = snapshot['volume_id']
        size_gb = snapshot['size_gb']
        
        # Calculate pricing per day
        price_per_gb_hour = PRICING['ebs_snapshot']
        price_per_day = price_per_gb_hour * size_gb * 24
        
        # Generate cost data points for each day (snapshots are billed continuously)
        current_time = start_time
        while current_time <= end_time:
            metrics.append({
                'timestamp': current_time.isoformat(),
                'resource_type': 'ebs_snapshot',
                'resource_id': snapshot_id,
                'metric_name': 'snapshot_cost',
                'value': price_per_day,  # Daily cost
                'unit': 'USD/day',
                'price_per_day': price_per_day,
                'size_gb': size_gb,
                'volume_id': volume_id,  # Link to parent volume
                'instance_type': '',
                'volume_type': ''
            })
            current_time += timedelta(days=1)
        
        return metrics
    
    def _get_load_balancer_metrics(self, lb: Dict, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get Load Balancer CloudWatch metrics."""
        metrics = []
        lb_name = lb['name']
        lb_type = lb['type']
        
        # Get pricing for this LB type (convert to daily)
        price_per_day = PRICING.get(lb_type, PRICING['alb']) * 24
        
        if lb_type in ['application', 'network']:
            # ALB/NLB metrics
            metric_queries = [
                ('RequestCount', 'AWS/ApplicationELB', 'Count'),
                ('TargetResponseTime', 'AWS/ApplicationELB', 'Seconds'),
                ('HTTPCode_Target_2XX_Count', 'AWS/ApplicationELB', 'Count'),
                ('HTTPCode_Target_4XX_Count', 'AWS/ApplicationELB', 'Count'),
                ('HTTPCode_Target_5XX_Count', 'AWS/ApplicationELB', 'Count'),
            ]
            
            if lb_type == 'network':
                # NLB specific metrics
                metric_queries = [
                    ('ActiveFlowCount_TCP', 'AWS/NetworkELB', 'Count'),
                    ('NewFlowCount_TCP', 'AWS/NetworkELB', 'Count'),
                    ('ProcessedBytes', 'AWS/NetworkELB', 'Bytes'),
                ]
        else:
            # Classic LB metrics
            metric_queries = [
                ('RequestCount', 'AWS/ELB', 'Count'),
                ('Latency', 'AWS/ELB', 'Seconds'),
                ('HTTPCode_Backend_2XX', 'AWS/ELB', 'Count'),
                ('HTTPCode_Backend_4XX', 'AWS/ELB', 'Count'),
                ('HTTPCode_Backend_5XX', 'AWS/ELB', 'Count'),
            ]
        
        for metric_name, namespace, unit in metric_queries:
            try:
                response = self.cloudwatch.get_metric_statistics(
                    Namespace=namespace,
                    MetricName=metric_name,
                    Dimensions=[
                        {
                            'Name': 'LoadBalancer' if lb_type == 'classic' else 'LoadBalancer',
                            'Value': lb_name
                        }
                    ],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=86400,  # 1 day
                    Statistics=['Sum'] if 'Count' in metric_name else ['Average']
                )
                
                for datapoint in response['Datapoints']:
                    value_key = 'Sum' if 'Count' in metric_name else 'Average'
                    metrics.append({
                        'timestamp': datapoint['Timestamp'].isoformat(),
                        'resource_type': 'load_balancer',
                        'resource_id': lb_name,
                        'lb_type': lb_type,
                        'metric_name': metric_name,
                        'value': datapoint[value_key],
                        'unit': unit,
                        'price_per_day': price_per_day
                    })
                    
            except Exception as e:
                logger.warning(f"Error getting {metric_name} for {lb_name}: {str(e)}")
        
        return metrics
    
    def save_to_csv(self, metrics_data: List[Dict], output_file: str):
        """Save metrics data to CSV file."""
        logger.info(f"Saving {len(metrics_data)} records to {output_file}")
        
        if not metrics_data:
            logger.warning("No data to save")
            return
        
        # Get all unique keys from all records
        all_keys = set()
        for record in metrics_data:
            all_keys.update(record.keys())
        
        fieldnames = sorted(list(all_keys))
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Sort by timestamp and resource_id for better organization
            sorted_data = sorted(metrics_data, key=lambda x: (x.get('timestamp', ''), x.get('resource_id', '')))
            writer.writerows(sorted_data)
        
        logger.info(f"Successfully saved data to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Collect EKS CloudWatch metrics and pricing data')
    parser.add_argument('--cluster-name', required=True, help='EKS cluster name')
    parser.add_argument('--profile', required=True, help='AWS profile name')
    parser.add_argument('--region', default='us-east-1', help='AWS region (default: us-east-1)')
    parser.add_argument('--days', type=int, default=30, help='Number of days to look back (default: 30)')
    parser.add_argument('--output', default='eks_metrics.csv', help='Output CSV file (default: eks_metrics.csv)')
    
    args = parser.parse_args()
    
    # Calculate time range
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=args.days)
    
    logger.info(f"Starting EKS CloudWatch collection for cluster: {args.cluster_name}")
    logger.info(f"Using AWS profile: {args.profile}")
    logger.info(f"Time range: {start_time} to {end_time}")
    
    try:
        # Initialize collector
        collector = EKSCloudWatchCollector(args.profile, args.region)
        
        # Discover resources
        resources = collector.get_cluster_resources(args.cluster_name)
        
        # Collect metrics
        metrics_data = collector.get_cloudwatch_metrics(resources, start_time, end_time)
        
        # Save to CSV
        collector.save_to_csv(metrics_data, args.output)
        
        logger.info("Collection completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during collection: {str(e)}")
        raise

if __name__ == '__main__':
    main()
