import java.io.BufferedReader; // Added this missing import
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class CustomerRiskDataGenerator {

    static final int RECORDS = 350000;

    public static void main(String[] args) {
        String inputPincodeFile = "pin_codes.csv";
        String outputFileName = "../assets/customer_risk_data.csv";
        Random rand = new Random();

        List<String> pincodeList = loadPincodes(inputPincodeFile);

        if (pincodeList.isEmpty()) {
            System.err.println("Error: No pin codes found. Please check your geo_analysis.csv file.");
            return;
        }

        try (FileWriter writer = new FileWriter(outputFileName)) {
            writer.append(
                "account_age_days,total_orders,return_rate,avg_order_value," +
                "cod_percentage,end_window_returns,damaged_claim_rate," +
                "linked_accounts,vpn_usage,past_fraud_flag,pin_code,customer_risk_label\n"
            );

            for (int i = 0; i < RECORDS; i++) {
                int accountAge      = rand.nextInt(1500) + 1;
                int totalOrders     = rand.nextInt(200)  + 1;
                double returnRate   = rand.nextDouble();
                double avgOrderVal  = 200 + rand.nextDouble() * 8000;
                double codPct       = rand.nextDouble();
                double endWindow    = rand.nextDouble();
                double damagedClaim = rand.nextDouble();
                int linkedAccounts  = rand.nextInt(6);
                int vpnUsage        = rand.nextInt(2);
                int pastFraud       = rand.nextDouble() < 0.08 ? 1 : 0;

                String randomPincode = pincodeList.get(rand.nextInt(pincodeList.size()));

                double riskScore = 0;
                if (accountAge < 90)           riskScore += 1.5;
                if (returnRate > 0.4)          riskScore += 2.0;
                if (codPct > 0.6)              riskScore += 1.0;
                if (endWindow > 0.5)           riskScore += 1.2;
                if (damagedClaim > 0.5)        riskScore += 1.3;
                if (linkedAccounts >= 3)       riskScore += 2.0;
                if (vpnUsage == 1)             riskScore += 1.0;
                if (pastFraud == 1)            riskScore += 3.0;

                double noise = rand.nextGaussian() * 1.5;
                double noisyScore = riskScore + noise;

                int customerRiskLabel = noisyScore >= 5 ? 1 : 0;

                writer.append(
                    accountAge + "," +
                    totalOrders + "," +
                    String.format("%.2f", returnRate) + "," +
                    String.format("%.2f", avgOrderVal) + "," +
                    String.format("%.2f", codPct) + "," +
                    String.format("%.2f", endWindow) + "," +
                    String.format("%.2f", damagedClaim) + "," +
                    linkedAccounts + "," +
                    vpnUsage + "," +
                    pastFraud + "," +
                    randomPincode + "," +
                    customerRiskLabel + "\n"
                );
            }
            System.out.println("✅ " + RECORDS + " customer risk records generated!");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static List<String> loadPincodes(String filePath) {
        List<String> list = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) list.add(line);
            }
        } catch (IOException e) {
            System.err.println("Could not read pincode file: " + e.getMessage());
        }
        return list;
    }
}
