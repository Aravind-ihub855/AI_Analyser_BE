# 📒 BP Dashboard & Business Rules Guide

**Document ID**: BP-BIZ-GUIDE-2026  
**Version**: 2.0  
**Effective Date**: March 1, 2026  
**Scope**: Dashboard metrics definitions, business logic, FSN parameters, and analytics rules.

---

## Section 1: Store Dashboard KPI Metric Calculations

### 1.1 Total Products
* **Definition**: The count of unique SKU product codes currently active in the store's inventory catalog.
* **Calculation**: Count of unique `code` keys in the inventory database table.

### 1.2 Inventory Valuation
* **Definition**: The total monetary value of current physical inventory at the store.
* **Calculation**:
  $$\text{Inventory Value} = \sum (\text{current\_stock} \times \text{unit\_price})$$
* **Dashboard Display**: The dashboard rounds and formats this value (e.g. $385,218.82 displayed as $385K or $39K based on category filters).

### 1.3 Days Left (Stock Days)
* **Definition**: Calculated timeline before an item runs out of stock.
* **Calculation**:
  $$\text{Days Left} = \lfloor \frac{\text{current\_stock}}{\text{avg\_daily\_consumption}} \rfloor$$

### 1.4 Critical Stockout Risk
* **Definition**: Any product with a Days Left metric <= 7 is marked as High Risk.
* **Overdue Orders Count**: Purchase orders where `expected_delivery_date` is in the past and status is not 'Delivered' or 'Cancelled'.

---

## Section 2: Fast, Slow, Non-Moving (FSN) Inventory Classification Rules

### 2.1 Fast-Moving (F)
* **Threshold**: Average daily consumption >= 25 units.
* **Action**: Weekly stock inspections. Requires immediate automatic reordering when hitting reorder points.

### 2.2 Slow-Moving (S)
* **Threshold**: Average daily consumption between 5 and 24 units.
* **Action**: Normal reordering intervals. Monthly review of safety stock levels.

### 2.3 Non-Moving (N)
* **Threshold**: Average daily consumption < 5 units.
* **Action**: Flagged for shelf-space reduction, potential discount promotion, or vendor returns.
