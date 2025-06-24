from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, RootModel


class BIOSVersion(RootModel):
    root: List[str]


class BIOS(BaseModel):
    BIOSVersion: BIOSVersion
    ReleaseDate: str
    SystemBiosMajorVersion: int
    SystemBiosMinorVersion: int


class NetworkAdapterItem(BaseModel):
    Name: str
    PNPDeviceID: Optional[str] = None
    PhysicalAdapter: bool
    AdapterType: Optional[str] = None


class NetworkAdapter(RootModel):
    root: List[NetworkAdapterItem]


class Communication(BaseModel):
    Network_adapter: NetworkAdapter = Field(alias="Network adapter")


class GraphicsCard(BaseModel):
    DriverVersion: str
    InstalledDisplayDrivers: str
    Name: str
    PNPDeviceID: str
    VideoModeDescription: str


class Monitor(BaseModel):
    Description: str
    Name: str
    PNPDeviceID: str


class Display(BaseModel):
    Graphics_card: GraphicsCard = Field(alias="Graphics card")
    Monitor: Monitor


class General(BaseModel):
    Manufacturer: str
    Model: str
    SystemFamily: str
    SystemSKUNumber: str
    SystemType: str


class PhysicalMemoryItem(BaseModel):
    BankLabel: str
    Capacity: str
    Speed: int


class PhysicalMemory(RootModel):
    root: List[PhysicalMemoryItem]


class PrinterItem(BaseModel):
    Name: str


class Printer(RootModel):
    root: List[PrinterItem]


class Processor(BaseModel):
    Architecture: int
    Description: str
    MaxClockSpeed: int
    Name: str
    NumberOfCores: int
    ThreadCount: int


class Drive(BaseModel):
    DeviceID: str
    InterfaceType: str
    Model: str
    Size: str


class LogicalDisk(BaseModel):
    FileSystem: str
    FreeSpace: str
    Name: str
    Size: str


class Storage(BaseModel):
    Drives: List[Drive]
    Logical_disks: List[LogicalDisk] = Field(alias="Logical disks")


class Hardware(BaseModel):
    BIOS: BIOS
    Communication: Communication
    Display: Display
    General: General
    Physical_memory: PhysicalMemory = Field(alias="Physical memory")
    Printer: Printer
    Processor: Processor
    Storage: Storage


class ApplicationCrash(BaseModel):
    appPath: str
    appVersion: str
    count: int
    exceptionCode: str
    faultOffset: str
    lastTime: datetime
    modulePath: str
    moduleVersion: str


class ApplicationHang(BaseModel):
    appPath: str
    appVersion: str
    count: int
    lastTime: datetime


class Driver(BaseModel):
    date: str
    name: str
    publisher: str
    version: str


class InstalledApplication(BaseModel):
    installDate: str
    name: str
    version: str


class OperatingSystem(BaseModel):
    BootDevice: str
    BuildNumber: str
    BuildType: str
    Caption: str
    CodeSet: str
    CountryCode: str
    CurrentTimeZone: int
    DataExecutionPrevention_32BitApplications: bool
    DataExecutionPrevention_Available: bool
    DataExecutionPrevention_Drivers: bool
    DataExecutionPrevention_SupportPolicy: int
    Debug: bool
    Description: Optional[str] = None
    Distributed: bool
    EncryptionLevel: int
    ForegroundApplicationBoost: int
    FreePhysicalMemory: str
    FreeSpaceInPagingFiles: str
    FreeVirtualMemory: str
    InstallDate: str
    LastBootUpTime: str
    LocalDateTime: str
    Locale: str
    MUILanguages: List[str]
    Manufacturer: str
    MaxNumberOfProcesses: int
    MaxProcessMemorySize: str
    Name: str
    NumberOfProcesses: int
    NumberOfUsers: int
    OSArchitecture: str
    OSLanguage: int
    OSProductSuite: int
    OSType: int
    OperatingSystemSKU: int
    PortableOperatingSystem: bool
    Primary: bool
    ProductType: int
    ServicePackMajorVersion: int
    ServicePackMinorVersion: int
    SizeStoredInPagingFiles: str
    Status: str
    SuiteMask: int
    SystemDevice: str
    SystemDirectory: str
    SystemDrive: str
    TotalVirtualMemorySize: str
    TotalVisibleMemorySize: str
    Version: str
    WindowsDirectory: str


class Service(BaseModel):
    Caption: str
    Name: str
    PathName: Optional[str] = None
    ServiceType: str
    StartMode: str
    State: str
    Status: str


class Software(BaseModel):
    Application_crashes: List[ApplicationCrash] = Field(alias="Application crashes")
    Application_hangs: List[ApplicationHang] = Field(alias="Application hangs")
    Drivers: List[Driver]
    Installed_applications: List[InstalledApplication] = Field(alias="Installed applications")
    Operating_system: OperatingSystem = Field(alias="Operating system")
    Services: List[Service]


class Root(BaseModel):
    Hardware: Hardware
    Software: Software