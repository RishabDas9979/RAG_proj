"""BankComply: a KYC/AML case agent with role-based access and a tamper-evident audit log."""

from .agent import CaseAgent
from .aml_rules import AmlConfig, Alert, run_rules
from .audit import AuditLog
from .cases import CaseService
from .data import Customer, Dataset, Transaction, WatchlistEntry, generate_dataset, load_paysim_csv
from .rbac import PermissionDenied, Principal

__version__ = "0.1.0"
__all__ = [
    "Alert", "AmlConfig", "AuditLog", "CaseAgent", "CaseService", "Customer", "Dataset",
    "PermissionDenied", "Principal", "Transaction", "WatchlistEntry", "generate_dataset",
    "load_paysim_csv", "run_rules",
]
