#!/usr/bin/env python3
"""
CSV Splitter for EKS CloudWatch Collector Output

Splits the main CSV output into separate files by resource type:
- ec2_instances.csv - EC2 instance metrics (CPU, disk I/O, costs)
- ebs_volumes.csv - EBS volume costs
- ebs_snapshots.csv - EBS snapshot costs  
- eks_clusters.csv - EKS cluster costs
- load_balancers.csv - Load balancer metrics
"""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

def split_csv_by_resource_type(input_file, output_dir=None):
    """
    Split CSV file by resource_type column into separate files.
    
    Args:
        input_file (str): Path to input CSV file
        output_dir (str): Directory to save split files (default: same as input)
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Error: Input file '{input_file}' not found")
        return False
    
    # Set output directory
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Resource type to filename mapping
    resource_files = {
        'ec2_instance': 'ec2_instances.csv',
        'ebs_volume': 'ebs_volumes.csv', 
        'ebs_snapshot': 'ebs_snapshots.csv',
        'eks_cluster': 'eks_clusters.csv',
        'load_balancer': 'load_balancers.csv'
    }
    
    # Store data by resource type
    resource_data = defaultdict(list)
    header = None
    
    print(f"Reading data from: {input_file}")
    
    # Read and categorize data
    with open(input_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        header = reader.fieldnames
        
        for row in reader:
            resource_type = row.get('resource_type', 'unknown')
            resource_data[resource_type].append(row)
    
    if not header:
        print("Error: Could not read CSV header")
        return False
    
    print(f"Found {len(resource_data)} resource types:")
    for resource_type, rows in resource_data.items():
        print(f"  - {resource_type}: {len(rows)} records")
    
    # Write separate files for each resource type
    files_created = []
    
    for resource_type, rows in resource_data.items():
        if not rows:
            continue
            
        # Get filename for this resource type
        filename = resource_files.get(resource_type, f"{resource_type}.csv")
        output_file = output_dir / filename
        
        print(f"Writing {len(rows)} records to: {output_file}")
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        
        files_created.append(str(output_file))
    
    print(f"\n✅ Successfully created {len(files_created)} files:")
    for file_path in sorted(files_created):
        file_size = os.path.getsize(file_path)
        print(f"  - {Path(file_path).name} ({file_size:,} bytes)")
    
    return True

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python split_csv_by_resource.py <input_csv> [output_directory]")
        print("\nExample:")
        print("  python split_csv_by_resource.py output_daily_pricing.csv")
        print("  python split_csv_by_resource.py output_daily_pricing.csv ./split_output/")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = split_csv_by_resource_type(input_file, output_dir)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

