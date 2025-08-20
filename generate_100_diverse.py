#!/usr/bin/env python3
"""
정확히 100개의 매우 다양한 Mermaid 파일 생성기
"""

import os
from pathlib import Path

def create_100_diverse_files():
    """정확히 100개의 다양한 Mermaid 파일 생성"""
    
    mermaid_dir = Path("mermaid")
    
    # 기존 diverse 파일들 정리
    for existing_file in mermaid_dir.glob("diverse*.mmd"):
        existing_file.unlink()
    
    files = []
    
    # 1-10: 소프트웨어 개발 프로세스
    dev_patterns = [
        """flowchart LR
    Code --> Git[Git Push]
    Git --> Build{Build?}
    Build -->|Pass| Test[Unit Tests]
    Build -->|Fail| Fix[Fix Bugs]
    Test --> Deploy[Deploy]""",
        
        """flowchart TD
    Client --> Gateway[API Gateway]
    Gateway --> Auth[Authentication]
    Gateway --> Service1[User Service]
    Gateway --> Service2[Order Service]
    Service1 --> DB1[(User DB)]
    Service2 --> DB2[(Order DB)]""",
        
        """flowchart LR
    Issue[Bug Report] --> Triage{Severity}
    Triage -->|High| Urgent[Urgent Fix]
    Triage -->|Medium| Planned[Plan Fix]
    Triage -->|Low| Backlog[Add to Backlog]
    Urgent --> Deploy
    Planned --> Sprint[Add to Sprint]""",
        
        """flowchart TD
    Develop --> PR[Pull Request]
    PR --> Review[Code Review]
    Review -->|Approve| Merge
    Review -->|Changes| Develop
    Merge --> CI[CI Pipeline]
    CI --> CD[CD Pipeline]""",
        
        """flowchart LR
    Requirements --> Design
    Design --> Implementation
    Implementation --> Testing
    Testing --> Deployment
    Deployment --> Monitoring
    Monitoring -->|Issues| Requirements""",
        
        """flowchart TD
    Sprint[Sprint Planning] --> Daily[Daily Standups]
    Daily --> Development
    Development --> Testing
    Testing --> Review[Sprint Review]
    Review --> Retro[Retrospective]
    Retro --> Sprint""",
        
        """flowchart LR
    Frontend --> API[REST API]
    API --> Backend[Backend Service]
    Backend --> Cache[Redis Cache]
    Backend --> Database[(PostgreSQL)]
    Cache -.-> Backend""",
        
        """flowchart TD
    User[User Story] --> Task[Development Task]
    Task --> Code[Write Code]
    Code --> Test[Write Tests]
    Test --> Review[Peer Review]
    Review --> Done[Definition of Done]""",
        
        """flowchart LR
    Local[Local Development] --> Stage[Staging Environment]
    Stage --> UAT[User Acceptance Testing]
    UAT --> Prod[Production]
    Prod --> Monitor[Monitoring & Alerts]""",
        
        """flowchart TD
    Feature[Feature Request] --> Analysis[Business Analysis]
    Analysis --> Design[Technical Design]
    Design --> Estimate[Time Estimation]
    Estimate --> Approval[Stakeholder Approval]
    Approval --> Development[Implementation]"""
    ]
    
    # 11-20: 비즈니스 프로세스
    business_patterns = [
        """flowchart LR
    Customer --> Browse[Browse Products]
    Browse --> Select[Select Item]
    Select --> Cart[Add to Cart]
    Cart --> Checkout[Checkout]
    Checkout --> Payment[Process Payment]
    Payment --> Ship[Shipping]""",
        
        """flowchart TD
    Lead[Sales Lead] --> Qualify{Qualify Lead}
    Qualify -->|Yes| Contact[Initial Contact]
    Qualify -->|No| Archive[Archive Lead]
    Contact --> Demo[Product Demo]
    Demo --> Proposal[Send Proposal]
    Proposal --> Close[Close Deal]""",
        
        """flowchart LR
    Application[Job Application] --> Screen[Resume Screening]
    Screen --> Phone[Phone Interview]
    Phone --> OnSite[Onsite Interview]
    OnSite --> Decision[Hiring Decision]
    Decision --> Offer[Job Offer]""",
        
        """flowchart TD
    Invoice[Create Invoice] --> Send[Send to Customer]
    Send --> Payment[Receive Payment]
    Payment --> Reconcile[Reconcile Accounts]
    Reconcile --> Report[Financial Report]""",
        
        """flowchart LR
    Order --> Inventory{Check Stock}
    Inventory -->|Available| Pick[Pick Items]
    Inventory -->|Out of Stock| Backorder[Create Backorder]
    Pick --> Pack[Package]
    Pack --> Ship[Ship Order]""",
        
        """flowchart TD
    Complaint[Customer Complaint] --> Investigate[Investigate Issue]
    Investigate --> Solution[Propose Solution]
    Solution --> Implement[Implement Fix]
    Implement --> Follow[Follow Up]
    Follow --> Close[Close Ticket]""",
        
        """flowchart LR
    Budget[Annual Budget] --> Department[Department Allocation]
    Department --> Project[Project Funding]
    Project --> Track[Expense Tracking]
    Track --> Report[Monthly Report]""",
        
        """flowchart TD
    Contract[Contract Draft] --> Legal[Legal Review]
    Legal --> Negotiate[Negotiation]
    Negotiate --> Sign[Digital Signature]
    Sign --> Execute[Contract Execution]""",
        
        """flowchart LR
    Employee --> Training[Training Program]
    Training --> Assessment[Skills Assessment]
    Assessment --> Certification[Get Certification]
    Certification --> Career[Career Development]""",
        
        """flowchart TD
    Product[New Product] --> Market[Market Research]
    Market --> Price[Pricing Strategy]
    Price --> Launch[Product Launch]
    Launch --> Monitor[Sales Monitoring]"""
    ]
    
    # 21-30: 시스템 아키텍처
    system_patterns = [
        """flowchart TB
    Users --> CDN[Content Delivery Network]
    CDN --> LB[Load Balancer]
    LB --> Web1[Web Server 1]
    LB --> Web2[Web Server 2]
    Web1 --> App[Application Server]
    Web2 --> App
    App --> DB[(Database)]""",
        
        """flowchart LR
    Source[(Source DB)] --> ETL[ETL Process]
    ETL --> Stage[(Staging)]
    Stage --> Transform[Data Transform]
    Transform --> Warehouse[(Data Warehouse)]
    Warehouse --> BI[Business Intelligence]""",
        
        """flowchart TD
    subgraph Docker["Docker Containers"]
        App1[App Container 1]
        App2[App Container 2]
        DB[DB Container]
    end
    LoadBalancer --> Docker
    App1 --> DB
    App2 --> DB""",
        
        """flowchart LR
    API[API Gateway] --> Auth[Auth Service]
    API --> User[User Service] 
    API --> Order[Order Service]
    User --> UserDB[(User DB)]
    Order --> OrderDB[(Order DB)]""",
        
        """flowchart TD
    Event[Event Source] --> Queue[Message Queue]
    Queue --> Consumer1[Consumer 1]
    Queue --> Consumer2[Consumer 2]
    Consumer1 --> Process1[Process A]
    Consumer2 --> Process2[Process B]""",
        
        """flowchart LR
    Client --> Proxy[Reverse Proxy]
    Proxy --> Cache[Cache Layer]
    Cache --> Backend[Backend Service]
    Backend --> Storage[(Data Storage)]""",
        
        """flowchart TD
    Monitor[Monitoring] --> Logs[Log Collection]
    Monitor --> Metrics[Metrics Collection]
    Logs --> Analysis[Log Analysis]
    Metrics --> Dashboard[Metrics Dashboard]
    Analysis --> Alerts[Alert System]""",
        
        """flowchart LR
    Upload[File Upload] --> Virus[Virus Scan]
    Virus --> Storage[Cloud Storage]
    Storage --> CDN[CDN Distribution]
    CDN --> Access[User Access]""",
        
        """flowchart TD
    Request[API Request] --> Validate[Input Validation]
    Validate --> Auth[Authorization]
    Auth --> Business[Business Logic]
    Business --> Data[Data Layer]
    Data --> Response[API Response]""",
        
        """flowchart LR
    Backup[Data Backup] --> Verify[Verify Backup]
    Verify --> Archive[Archive Storage]
    Archive --> Restore[Disaster Recovery]
    Restore --> Validate[Validate Restore]"""
    ]
    
    # 31-40: 교육 시스템
    education_patterns = [
        """flowchart TD
    Student --> Enroll[Course Enrollment]
    Enroll --> Lesson[Attend Lessons]
    Lesson --> Quiz[Take Quiz]
    Quiz --> Assignment[Submit Assignment]
    Assignment --> Grade[Receive Grade]""",
        
        """flowchart LR
    Research[Research Topic] --> Proposal[Write Proposal]
    Proposal --> Advisor[Advisor Review]
    Advisor --> Experiment[Conduct Experiment]
    Experiment --> Analysis[Data Analysis]
    Analysis --> Paper[Write Paper]""",
        
        """flowchart TD
    Library[Library System] --> Search[Search Books]
    Search --> Reserve[Reserve Book]
    Reserve --> Checkout[Check Out]
    Checkout --> Return[Return Book]
    Return --> Fine{Late Fee?}""",
        
        """flowchart LR
    Exam[Create Exam] --> Schedule[Schedule Exam]
    Schedule --> Proctor[Proctor Exam]
    Proctor --> Grade[Grade Papers]
    Grade --> Results[Publish Results]""",
        
        """flowchart TD
    Curriculum[Design Curriculum] --> Content[Create Content]
    Content --> Delivery[Content Delivery]
    Delivery --> Feedback[Student Feedback]
    Feedback --> Improve[Curriculum Improvement]""",
        
        """flowchart LR
    Application[Admission Application] --> Review[Application Review]
    Review --> Interview[Student Interview]
    Interview --> Decision[Admission Decision]
    Decision --> Enroll[Student Enrollment]""",
        
        """flowchart TD
    Lab[Lab Session] --> Safety[Safety Briefing]
    Safety --> Experiment[Conduct Experiment]
    Experiment --> Record[Record Data]
    Record --> Report[Lab Report]""",
        
        """flowchart LR
    Degree[Degree Requirements] --> Credits[Credit Tracking]
    Credits --> Advisor[Academic Advising]
    Advisor --> Planning[Course Planning]
    Planning --> Graduation[Graduation Check]""",
        
        """flowchart TD
    Workshop[Workshop] --> Registration[Student Registration]
    Registration --> Materials[Prepare Materials]
    Materials --> Session[Conduct Session]
    Session --> Certificate[Issue Certificate]""",
        
        """flowchart LR
    Assessment[Student Assessment] --> Rubric[Apply Rubric]
    Rubric --> Score[Calculate Score]
    Score --> Feedback[Provide Feedback]
    Feedback --> Improvement[Learning Improvement]"""
    ]
    
    # 41-50: 게임 시스템
    game_patterns = [
        """flowchart TD
    Player --> Character[Create Character]
    Character --> Tutorial[Play Tutorial]
    Tutorial --> World[Enter Game World]
    World --> Quest[Accept Quest]
    Quest --> Combat[Combat System]
    Combat --> Reward[Earn Reward]""",
        
        """flowchart LR
    Match[Find Match] --> Lobby[Game Lobby]
    Lobby --> Start[Start Game]
    Start --> Play[Gameplay]
    Play --> End[Game End]
    End --> Stats[Show Statistics]""",
        
        """flowchart TD
    Achievement[Achievement System] --> Progress[Track Progress]
    Progress --> Complete[Achievement Complete]
    Complete --> Unlock[Unlock Reward]
    Unlock --> Display[Display Badge]""",
        
        """flowchart LR
    Item[Collect Item] --> Inventory[Add to Inventory]
    Inventory --> Craft[Crafting System]
    Craft --> Upgrade[Item Upgrade]
    Upgrade --> Equip[Equip Item]""",
        
        """flowchart TD
    Guild[Join Guild] --> Member[Guild Member]
    Member --> Event[Guild Event]
    Event --> Contribute[Contribute Points]
    Contribute --> Rank[Guild Ranking]""",
        
        """flowchart LR
    Battle[PvP Battle] --> Matchmaking[Player Matching]
    Matchmaking --> Arena[Battle Arena]
    Arena --> Victory[Declare Winner]
    Victory --> Rating[Update Rating]""",
        
        """flowchart TD
    Economy[Game Economy] --> Shop[In-Game Shop]
    Shop --> Purchase[Buy Items]
    Purchase --> Currency[Virtual Currency]
    Currency --> Trade[Player Trading]""",
        
        """flowchart LR
    Level[Level Design] --> Test[Playtest]
    Test --> Balance[Game Balance]
    Balance --> Deploy[Level Release]
    Deploy --> Feedback[Player Feedback]""",
        
        """flowchart TD
    Tournament[Tournament] --> Bracket[Create Bracket]
    Bracket --> Rounds[Tournament Rounds]
    Rounds --> Finals[Grand Finals]
    Finals --> Champion[Declare Champion]""",
        
        """flowchart LR
    Mod[Game Mod] --> Upload[Upload to Workshop]
    Upload --> Review[Community Review]
    Review --> Feature[Featured Mod]
    Feature --> Download[Player Downloads]"""
    ]
    
    # 51-60: 의료 시스템
    healthcare_patterns = [
        """flowchart TD
    Patient --> Registration[Patient Registration]
    Registration --> Triage[Medical Triage]
    Triage --> Doctor[See Doctor]
    Doctor --> Diagnosis[Medical Diagnosis]
    Diagnosis --> Treatment[Treatment Plan]""",
        
        """flowchart LR
    Symptom[Report Symptoms] --> Examination[Physical Exam]
    Examination --> Tests[Medical Tests]
    Tests --> Results[Test Results]
    Results --> Prescription[Write Prescription]""",
        
        """flowchart TD
    Surgery[Schedule Surgery] --> Prep[Pre-Op Preparation]
    Prep --> OR[Operating Room]
    OR --> Recovery[Recovery Room]
    Recovery --> Discharge[Patient Discharge]""",
        
        """flowchart LR
    Emergency[Emergency Call] --> Dispatch[Dispatch Ambulance]
    Dispatch --> Scene[Arrive at Scene]
    Scene --> Stabilize[Stabilize Patient]
    Stabilize --> Transport[Transport to Hospital]""",
        
        """flowchart TD
    Pharmacy[Pharmacy] --> Prescription[Receive Prescription]
    Prescription --> Verify[Verify Medication]
    Verify --> Dispense[Dispense Medicine]
    Dispense --> Counseling[Patient Counseling]""",
        
        """flowchart LR
    Lab[Laboratory] --> Sample[Collect Sample]
    Sample --> Analysis[Lab Analysis]
    Analysis --> Report[Generate Report]
    Report --> Doctor[Send to Doctor]""",
        
        """flowchart TD
    Insurance[Insurance Claim] --> Verify[Verify Coverage]
    Verify --> Process[Process Claim]
    Process --> Approve[Approve Payment]
    Approve --> Reimburse[Reimburse Provider]""",
        
        """flowchart LR
    Radiology[Radiology] --> Scan[Medical Scan]
    Scan --> Image[Capture Image]
    Image --> Radiologist[Radiologist Review]
    Radiologist --> Report[Diagnostic Report]""",
        
        """flowchart TD
    Therapy[Physical Therapy] --> Assessment[Initial Assessment]
    Assessment --> Plan[Treatment Plan]
    Plan --> Session[Therapy Session]
    Session --> Progress[Track Progress]""",
        
        """flowchart LR
    Vaccine[Vaccination Program] --> Schedule[Schedule Appointment]
    Schedule --> Administer[Administer Vaccine]
    Administer --> Record[Update Records]
    Record --> Monitor[Monitor Side Effects]"""
    ]
    
    # 61-70: 제조 시스템
    manufacturing_patterns = [
        """flowchart TD
    Design --> Prototype[Build Prototype]
    Prototype --> Test[Product Testing]
    Test --> Production[Mass Production]
    Production --> Quality[Quality Control]
    Quality --> Ship[Ship Products]""",
        
        """flowchart LR
    Material[Raw Materials] --> Inventory[Material Inventory]
    Inventory --> Production[Production Line]
    Production --> Assembly[Product Assembly]
    Assembly --> Packaging[Product Packaging]""",
        
        """flowchart TD
    Order[Customer Order] --> Planning[Production Planning]
    Planning --> Schedule[Production Schedule]
    Schedule --> Manufacturing[Manufacturing Process]
    Manufacturing --> Delivery[Product Delivery]""",
        
        """flowchart LR
    Machine[Manufacturing Machine] --> Operation[Machine Operation]
    Operation --> Monitor[Performance Monitor]
    Monitor --> Maintenance[Preventive Maintenance]
    Maintenance --> Operation""",
        
        """flowchart TD
    Inspection[Quality Inspection] --> Standard{Meet Standards?}
    Standard -->|Yes| Accept[Accept Product]
    Standard -->|No| Reject[Reject Product]
    Reject --> Rework[Product Rework]
    Rework --> Inspection""",
        
        """flowchart LR
    Supply[Supply Chain] --> Vendor[Vendor Management]
    Vendor --> Purchase[Purchase Order]
    Purchase --> Receive[Receive Goods]
    Receive --> Storage[Warehouse Storage]""",
        
        """flowchart TD
    Safety[Safety Protocol] --> Training[Safety Training]
    Training --> Equipment[Safety Equipment]
    Equipment --> Compliance[Safety Compliance]
    Compliance --> Audit[Safety Audit]""",
        
        """flowchart LR
    Waste[Manufacturing Waste] --> Sort[Waste Sorting]
    Sort --> Recycle[Recycling Process]
    Recycle --> Reuse[Material Reuse]
    Reuse --> Reduce[Waste Reduction]""",
        
        """flowchart TD
    Lean[Lean Manufacturing] --> Analyze[Process Analysis]
    Analyze --> Improve[Process Improvement]
    Improve --> Implement[Implementation]
    Implement --> Monitor[Performance Monitor]""",
        
        """flowchart LR
    Robot[Industrial Robot] --> Program[Robot Programming]
    Program --> Calibrate[Robot Calibration]
    Calibrate --> Automate[Automated Process]
    Automate --> Efficiency[Improved Efficiency]"""
    ]
    
    # 71-80: 금융 시스템
    finance_patterns = [
        """flowchart TD
    Application[Loan Application] --> Verify[Document Verification]
    Verify --> Credit[Credit Check]
    Credit --> Approval{Loan Approval}
    Approval -->|Yes| Disburse[Loan Disbursement]
    Approval -->|No| Decline[Application Declined]""",
        
        """flowchart LR
    Investment[Investment Account] --> Portfolio[Build Portfolio]
    Portfolio --> Trade[Execute Trades]
    Trade --> Monitor[Monitor Performance]
    Monitor --> Rebalance[Portfolio Rebalancing]""",
        
        """flowchart TD
    Transaction[Bank Transaction] --> Validate[Validate Transaction]
    Validate --> Process[Process Payment]
    Process --> Settlement[Transaction Settlement]
    Settlement --> Record[Update Records]""",
        
        """flowchart LR
    Fraud[Fraud Detection] --> Alert[Generate Alert]
    Alert --> Investigate[Investigation]
    Investigate --> Action[Take Action]
    Action --> Report[Fraud Report]""",
        
        """flowchart TD
    Budget[Create Budget] --> Track[Expense Tracking]
    Track --> Compare[Budget vs Actual]
    Compare --> Adjust[Budget Adjustment]
    Adjust --> Report[Financial Report]""",
        
        """flowchart LR
    Insurance[Insurance Policy] --> Premium[Premium Payment]
    Premium --> Claim[File Claim]
    Claim --> Assess[Claim Assessment]
    Assess --> Payout[Insurance Payout]""",
        
        """flowchart TD
    Mortgage[Mortgage Application] --> Appraisal[Property Appraisal]
    Appraisal --> Underwriting[Loan Underwriting]
    Underwriting --> Closing[Loan Closing]
    Closing --> Service[Loan Servicing]""",
        
        """flowchart LR
    Risk[Risk Assessment] --> Model[Risk Model]
    Model --> Score[Risk Score]
    Score --> Decision[Decision Making]
    Decision --> Monitor[Risk Monitoring]""",
        
        """flowchart TD
    Compliance[Regulatory Compliance] --> Audit[Internal Audit]
    Audit --> Report[Compliance Report]
    Report --> Regulator[Submit to Regulator]
    Regulator --> Feedback[Regulatory Feedback]""",
        
        """flowchart LR
    Payment[Digital Payment] --> Gateway[Payment Gateway]
    Gateway --> Processor[Payment Processor]
    Processor --> Bank[Banking Network]
    Bank --> Confirmation[Payment Confirmation]"""
    ]
    
    # 81-90: 물류 시스템
    logistics_patterns = [
        """flowchart TD
    Order --> Warehouse[Warehouse Pick]
    Warehouse --> Pack[Package Item]
    Pack --> Label[Shipping Label]
    Label --> Carrier[Shipping Carrier]
    Carrier --> Delivery[Final Delivery]""",
        
        """flowchart LR
    Supplier --> Receive[Receive Goods]
    Receive --> Inspect[Quality Inspection]
    Inspect --> Store[Store in Warehouse]
    Store --> Distribute[Distribution Center]""",
        
        """flowchart TD
    Route[Route Planning] --> Optimize[Route Optimization]
    Optimize --> Dispatch[Dispatch Vehicle]
    Dispatch --> Track[GPS Tracking]
    Track --> Arrive[Arrival Confirmation]""",
        
        """flowchart LR
    Inventory --> Count[Inventory Count]
    Count --> Reconcile[Reconcile Differences]
    Reconcile --> Adjust[Inventory Adjustment]
    Adjust --> Report[Inventory Report]""",
        
        """flowchart TD
    Import[Import Goods] --> Customs[Customs Clearance]
    Customs --> Documentation[Import Documentation]
    Documentation --> Release[Goods Release]
    Release --> Domestic[Domestic Distribution]""",
        
        """flowchart LR
    Fleet[Fleet Management] --> Vehicle[Vehicle Maintenance]
    Vehicle --> Driver[Driver Assignment]
    Driver --> Route[Route Assignment]
    Route --> Performance[Performance Tracking]""",
        
        """flowchart TD
    Return[Product Return] --> Authorize[Return Authorization]
    Authorize --> Inspect[Return Inspection]
    Inspect --> Refund[Process Refund]
    Refund --> Restock[Restock Item]""",
        
        """flowchart LR
    Cold[Cold Chain] --> Temperature[Temperature Monitor]
    Temperature --> Transport[Refrigerated Transport]
    Transport --> Verify[Temperature Verification]
    Verify --> Deliver[Cold Delivery]""",
        
        """flowchart TD
    Forecast[Demand Forecast] --> Plan[Supply Planning]
    Plan --> Procure[Procurement]
    Procure --> Stock[Stock Management]
    Stock --> Fulfill[Order Fulfillment]""",
        
        """flowchart LR
    Cross[Cross Dock] --> Unload[Unload Truck]
    Unload --> Sort[Sort Products]
    Sort --> Load[Load Truck]
    Load --> Ship[Ship Out]"""
    ]
    
    # 91-100: 특별한 패턴들
    special_patterns = [
        """flowchart TD
    subgraph A["데이터 입력"]
        Input1[입력 A]
        Input2[입력 B]
    end
    subgraph B["데이터 처리"]
        Process1[처리 A]
        Process2[처리 B]
    end
    subgraph C["결과 출력"]
        Output1[출력 A]
        Output2[출력 B]
    end
    A --> B
    B --> C""",
        
        """flowchart LR
    Start --> Decision1{조건 1}
    Decision1 -->|True| Decision2{조건 2}
    Decision1 -->|False| End1[종료 A]
    Decision2 -->|True| Action1[액션 A]
    Decision2 -->|False| Action2[액션 B]
    Action1 --> End2[종료 B]
    Action2 --> End2""",
        
        """flowchart TD
    Alpha[시스템 알파] -.-> Beta[시스템 베타]
    Beta === Gamma[시스템 감마]
    Gamma --> Delta[시스템 델타]
    Delta <--> Epsilon[시스템 엡실론]
    Alpha --> Omega[시스템 오메가]""",
        
        """flowchart LR
    Node1 --> Node2
    Node2 --> Node3
    Node3 --> Node4
    Node4 --> Node5
    Node5 --> Node6
    Node6 --> Node7
    Node7 --> Node8
    Node8 --> Node1""",
        
        """flowchart TD
    Central[중앙 노드] --> Branch1[분기 1]
    Central --> Branch2[분기 2]
    Central --> Branch3[분기 3]
    Branch1 --> Leaf1[리프 1]
    Branch1 --> Leaf2[리프 2]
    Branch2 --> Leaf3[리프 3]
    Branch3 --> Leaf4[리프 4]""",
        
        """flowchart LR
    Layer1[계층 1] --> Layer2[계층 2]
    Layer2 --> Layer3[계층 3]
    Layer3 --> Layer4[계층 4]
    Layer4 --> Layer5[계층 5]
    Layer5 -.-> Layer1""",
        
        """flowchart TD
    Matrix1[매트릭스 A] & Matrix2[매트릭스 B] --> Compute[연산 처리]
    Compute --> Result1[결과 A] & Result2[결과 B]
    Result1 --> Output[최종 출력]
    Result2 --> Output""",
        
        """flowchart LR
    Parallel1[병렬 처리 1] --> Sync[동기화 지점]
    Parallel2[병렬 처리 2] --> Sync
    Parallel3[병렬 처리 3] --> Sync
    Sync --> Merge[병합 처리]""",
        
        """flowchart TD
    Pipeline1[파이프라인 1단계] --> Pipeline2[파이프라인 2단계]
    Pipeline2 --> Pipeline3[파이프라인 3단계]
    Pipeline3 --> Pipeline4[파이프라인 4단계]
    Pipeline4 --> Pipeline5[파이프라인 5단계]""",
        
        """flowchart LR
    Hub[허브] --> Spoke1[스포크 1]
    Hub --> Spoke2[스포크 2]
    Hub --> Spoke3[스포크 3]
    Hub --> Spoke4[스포크 4]
    Spoke1 -.-> Spoke2
    Spoke2 -.-> Spoke3
    Spoke3 -.-> Spoke4"""
    ]
    
    # 모든 패턴 합치기
    all_patterns = (dev_patterns + business_patterns + system_patterns + 
                   education_patterns + game_patterns + healthcare_patterns +
                   manufacturing_patterns + finance_patterns + logistics_patterns + 
                   special_patterns)
    
    # 파일 생성
    for i, pattern in enumerate(all_patterns, 1):
        filename = f"diverse{i:03d}.mmd"
        filepath = mermaid_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(pattern)
        
        files.append(filename)
        print(f"생성됨: {filename}")
    
    print(f"\n총 {len(files)}개의 매우 다양한 파일 생성 완료!")
    print("\n포함된 카테고리:")
    print("1-10: 소프트웨어 개발 프로세스")
    print("11-20: 비즈니스 프로세스") 
    print("21-30: 시스템 아키텍처")
    print("31-40: 교육 시스템")
    print("41-50: 게임 시스템")
    print("51-60: 의료 시스템")
    print("61-70: 제조 시스템")
    print("71-80: 금융 시스템")
    print("81-90: 물류 시스템")
    print("91-100: 특별한 패턴들")
    
    return len(files)

if __name__ == "__main__":
    count = create_100_diverse_files()
    print(f"\n완벽한 다양성 테스트 준비! {count}개의 실제적이고 다양한 파일로 level.py를 테스트하세요!")